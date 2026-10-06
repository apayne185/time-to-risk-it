# ADR 0008: AWS deployment on ECS Fargate

- **Status:** accepted (templates validated offline; not yet deployed to a live account)
- **Date:** 2026-10-05

## Context

The scoring service (ADR 0006) is a stateless container that needs one artifact, the model
bundle. It should run where an operator's engineering team already runs services, with no
long-lived credentials and a safe path to update both code and model.

## Decisions

1. **ECS Fargate behind an ALB.** No servers or cluster to manage, rolling deployments with a
   circuit breaker, and autoscaling on CPU. EKS would add a control plane and Kubernetes
   manifests for a single small service; Lambda would mean a cold-start-sensitive container with
   XGBoost and pandas.
2. **Model bundle in S3, fetched at startup.** The image holds code only (the bundle is trained on
   licensed data). The bucket is private, encrypted, versioned and TLS-only; the task role can read
   `bundles/*` and nothing else. Updating the model is an upload plus a forced redeployment;
   rolling back is restoring an earlier object version.
3. **Readiness gate.** The ALB health check uses `/ready`, which returns 503 until the bundle is
   loaded. A bad or missing bundle keeps new tasks out of rotation and the circuit breaker rolls
   the deployment back, instead of serving errors.
4. **Three stacks.** Foundation (registry, bucket) changes rarely and is retained; deployment access
   is separate so its IAM can be reviewed on its own; the service stack is what CI redeploys.
5. **GitHub OIDC, two roles.** The workflow assumes a role scoped to one repository environment,
   which can push to one ECR repository and update one stack. CloudFormation runs as a separate
   execution role, so the CI identity never holds the permissions to create infrastructure
   directly.
6. **Manual, approved deploys.** `workflow_dispatch` in a `production` environment with required
   reviewers. CI never deploys.

## Consequences

- Templates are linted in CI (`cfn-lint`) and S3 loading is tested with moto, but behaviour in a
  real account (IAM edge cases, networking) is unverified until a first deployment.
- The monthly recalibration (`ttr monitor --write-shift`) runs outside AWS for now; a scheduled
  ECS task with access to the feature table is the natural next step.
- Running cost is roughly $60/month for a two-task demo in public subnets, more with a NAT
  gateway (see `docs/deploy-aws.md`).
