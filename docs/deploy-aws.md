# Deploying the scoring service to AWS

The scoring API runs on **ECS Fargate behind an Application Load Balancer**, pulling its model
bundle from a private S3 bucket. Infrastructure is three CloudFormation stacks in
[`infra/aws/`](../infra/aws); deployments go through a manual GitHub Actions workflow using OIDC,
so no AWS keys are stored anywhere.

> **Status:** the templates pass `cfn-lint` in CI and the S3 bundle loading is tested against
> moto, but this has **not been deployed to a live AWS account**. Treat the first deployment as
> a test and watch the CloudFormation events.

```
GitHub Actions (OIDC) ──push──▶ ECR ──image──▶ ECS Fargate tasks ──/ready──▶ ALB ──▶ RG tools
                                                    │
                                     s3://…/bundles/primary/decision_model.joblib
```

| Stack | Template | Contents |
|---|---|---|
| `ttr-foundation` | `foundation.yaml` | ECR repository (immutable, scanned tags), model bucket (private, encrypted, versioned, TLS-only, retained on delete) |
| `ttr-github-deploy` | `github-deploy.yaml` | GitHub OIDC provider, deploy role (ECR push + this stack only), CloudFormation execution role |
| `ttr-scoring-service` | `scoring-service.yaml` | ECS cluster and service, ALB (HTTPS optional), autoscaling, logs, alarms, least-privilege task role |

## Prerequisites

- AWS CLI v2 with an admin-capable profile for the one-off setup.
- A VPC with two public subnets in different AZs for the load balancer. The default VPC works.
- For the tasks: private subnets with a NAT gateway (recommended), or the public subnets with
  `AssignPublicIp=ENABLED` (cheaper demo).
- Optional: an ACM certificate for HTTPS.

```bash
export AWS_REGION=eu-west-1 PROJECT=ttr
```

## 1. Foundation (once)

```bash
aws cloudformation deploy --stack-name $PROJECT-foundation \
  --template-file infra/aws/foundation.yaml --parameter-overrides ProjectName=$PROJECT
BUCKET=$(aws cloudformation describe-stacks --stack-name $PROJECT-foundation \
  --query "Stacks[0].Outputs[?OutputKey=='ModelBucketName'].OutputValue" --output text)
```

## 2. Upload a model bundle

The real-data bundle is trained on licensed data, so it lives only in this private bucket. To try
the stack without the licence, upload the synthetic model from `make demo` instead.

```bash
make evaluate                                   # or: make demo
aws s3 cp models/primary/decision_model.joblib \
  s3://$BUCKET/bundles/primary/decision_model.joblib
```

## 3. Deployment access (once)

```bash
aws cloudformation deploy --stack-name $PROJECT-github-deploy \
  --template-file infra/aws/github-deploy.yaml --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides ProjectName=$PROJECT GitHubRepository=apayne185/time-to-risk-it
# Set CreateOidcProvider=false if the account already has the GitHub OIDC provider.
aws cloudformation describe-stacks --stack-name $PROJECT-github-deploy --query "Stacks[0].Outputs"
```

In GitHub: **Settings → Environments → New environment `production`**. Add required reviewers
(so every deploy needs an approval), then:

| Type | Name | Value |
|---|---|---|
| Secret | `AWS_DEPLOY_ROLE_ARN` | `GitHubDeployRoleArn` output |
| Secret | `AWS_CFN_EXECUTION_ROLE_ARN` | `CloudFormationExecutionRoleArn` output |
| Variable | `AWS_REGION` | e.g. `eu-west-1` |
| Variable | `VPC_ID` | `vpc-…` |
| Variable | `LB_SUBNET_IDS` | comma-separated public subnets, e.g. `subnet-a,subnet-b` |
| Variable | `SERVICE_SUBNET_IDS` | comma-separated task subnets |
| Variable | `ASSIGN_PUBLIC_IP` | `ENABLED` only for public task subnets without NAT |
| Variable | `CERTIFICATE_ARN` | optional, enables HTTPS |
| Variable | `ALLOWED_INGRESS_CIDR` | optional, restrict who can reach the API |

## 4. Deploy

**Actions → deploy-aws → Run workflow.** It builds the image tagged with the commit SHA, pushes it
to ECR, deploys `ttr-scoring-service`, waits for ECS to stabilise and smoke-tests `/ready` and
`/model`. The service URL is in the run summary.

Rollouts are rolling (100% minimum healthy). Tasks only receive traffic once `/ready` reports a
loaded model, and the deployment circuit breaker rolls back a release whose tasks never become
healthy.

## Operating it

- **Monthly recalibration** (ADR 0005): run `ttr monitor --write-shift` once last month's outcomes
  are in, upload the updated bundle to the same key (the bucket keeps every version), then
  `aws ecs update-service --cluster $PROJECT-cluster --service $PROJECT-scoring-service
  --force-new-deployment`.
- **Rollback a model:** copy an earlier object version back to the key, or point
  `ModelBundleKey` at a different key and redeploy.
- **Logs:** CloudWatch log group `/ecs/ttr/scoring-service`. **Alarms:** target 5xx and
  unhealthy targets go to the `ttr-scoring-alarms` SNS topic (`AlarmEmail` parameter subscribes
  an address).

## Cost (rough, eu-west-1, check current pricing)

| Item | Monthly |
|---|---|
| 2 Fargate tasks, 0.5 vCPU / 1 GB, always on | ~$36 |
| Application Load Balancer (idle-to-light traffic) | ~$18–25 |
| CloudWatch logs, ECR, S3 | a few dollars |
| NAT gateway (only if tasks are in private subnets) | ~$35 + data |

Roughly **$60/month** for a public-subnet demo. Delete the stacks when you are done.

## Teardown

```bash
aws cloudformation delete-stack --stack-name $PROJECT-scoring-service
aws cloudformation wait stack-delete-complete --stack-name $PROJECT-scoring-service
aws cloudformation delete-stack --stack-name $PROJECT-github-deploy
aws ecr delete-repository --repository-name $PROJECT/scoring-service --force
aws cloudformation delete-stack --stack-name $PROJECT-foundation
# The model bucket is retained on purpose (licensed-data artifacts). Remove it explicitly:
aws s3 rb s3://$BUCKET --force   # versioned: delete object versions first if this fails
```
