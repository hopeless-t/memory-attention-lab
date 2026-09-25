# BENCH-001B — Google Cloud Setup Runbook

> **Status:** HUMAN SETUP CHECKPOINT  
> **Performance authority:** NONE  
> **Paid-resource authority:** NONE  
> **GPU execution authority:** NONE

## Purpose

Prepare a Google Cloud project so BENCH-001B can later probe an A2 Standard
instance with one NVIDIA A100 40 GB GPU in Tokyo.

This runbook intentionally stops **before VM creation**.

Completing these steps should not start a GPU instance.

## Account assumptions

Use the same personal Google account for:

~~~text
Google AI Pro
Google Developer Program
Google Cloud Billing
Google Cloud project ownership
~~~

Google AI Pro currently includes the Google Developer Program Premium benefits
for personal accounts, including **$10 of Google Cloud credit per month**.

Google states that these Cloud credits can be applied to Google Cloud products,
including Compute Engine.

## Step 1 — Apply the monthly Google Cloud credit

Open the Google Developer Program dashboard and go to:

~~~text
Benefits
-> GenAI & Cloud developer credits
~~~

Choose the Cloud Billing account that should receive the benefit.

If the benefit card shows a promotion code instead of a billing-account selector:

~~~text
Google Cloud Console
-> Billing
-> Redeem promotion
~~~

and redeem the code against the intended billing account.

Do not post the billing-account ID, payment details, or promotion code into this
repository.

### Acceptance check

Before continuing, the Developer Program benefits page should show the benefit
as redeemed/applied to the intended billing account.

## Step 2 — Create a dedicated Cloud project

Recommended project name:

~~~text
memory-attention-lab-bench001b
~~~

A generated project ID is fine.

Link this project to the billing account that has the Google AI Pro / Developer
Program Cloud credit.

Keep this project separate from unrelated production workloads.

### Acceptance check

The Google Cloud Console project selector shows the new project and Billing
shows it linked to the intended billing account.

## Step 3 — Create a gross-spend budget alert

Open:

~~~text
Billing
-> Budgets & alerts
-> Create budget
~~~

Recommended configuration:

~~~text
scope:
    only memory-attention-lab-bench001b

budget type:
    Alerts only

budget amount:
    USD 10 monthly

thresholds:
    25%
    50%
    75%
    90%
    100%
~~~

Important:

Google Cloud budgets are alerts, not hard spending caps.

By default, budgets can account for credits and discounts. For this research
project, configure the budget to monitor **gross cost before promotional
credits** by excluding Savings/credits from the budget calculation when the UI
offers that option.

This makes alerts fire based on resource usage even when the monthly $10 credit
is masking the net bill.

### Acceptance check

A project-scoped monthly budget exists and email alerts are enabled.

## Step 4 — Enable Compute Engine

With the research project selected:

~~~text
APIs & Services
-> Library
-> Compute Engine API
-> Enable
~~~

Do not create a VM yet.

### Acceptance check

Compute Engine API status is Enabled.

## Step 5 — Request only the minimum A100 quota

The canonical Tokyo target is:

~~~text
region:
    asia-northeast1

zones:
    asia-northeast1-a
    asia-northeast1-c

machine family:
    A2 Standard

machine type:
    a2-highgpu-1g

GPU:
    NVIDIA A100 40 GB

GPU count:
    1
~~~

Google currently lists A2 Standard in Tokyo zones asia-northeast1-a and
asia-northeast1-c.

Open:

~~~text
IAM & Admin
-> Quotas & System Limits
~~~

Filter for Compute Engine GPU quotas.

Request only:

~~~text
NVIDIA A100 GPUs
    region: asia-northeast1
    requested value: 1

GPUs (all regions)
    requested value: at least 1
~~~

For A2 VMs, Google states that the A100 GPU quota is the required quota; a
separate A2 CPU quota request is not required.

Do **not** request A100 80 GB quota during this first setup.

Do **not** request RTX PRO 6000 quota for canonical BENCH-001B.

### Acceptance check

Quota request status for one standard A100 in asia-northeast1 is approved, or
the project clearly shows the existing quota is already >= 1.

## Step 6 — Stop

Do not create a2-highgpu-1g yet.

At this checkpoint the project should have:

~~~text
Google AI Pro / Developer Program credit:
    applied to billing account

dedicated project:
    created and linked to billing

gross-cost budget:
    enabled

Compute Engine API:
    enabled

A100 40 GB quota:
    1 requested / approved in asia-northeast1

GPU VM:
    NOT CREATED
~~~

Return to the BENCH-001B workflow for review before creating any billed GPU
resource.

## Why A2 Standard first

The current canonical BENCH-001B protocol uses the pinned upstream
FlashAttention-2 path.

A100 is Ampere and remains inside the official FA2 support envelope.

The Tokyo GPU-location table currently lists A2 Standard, but not A2 Ultra, in
asia-northeast1-a and asia-northeast1-c.

Therefore the first Google canonical probe is:

~~~text
A2 Standard
A100 40 GB
1 GPU
Tokyo
~~~

A2 Ultra / A100 80 GB remains a non-Tokyo headroom option if later needed.

## Security and automation boundary

Cloud Build is not required for this setup.

Do not grant Cloud Build or any service account permission to create GPU VMs
during this checkpoint.

A future control-plane automation may create and delete a disposable A2 VM only
after a separate reviewed decision that freezes:

~~~text
service account
IAM roles
budget ceiling
VM lifetime
cleanup guarantee
failure cleanup
evidence retrieval
~~~

## Information safe to report back

Safe:

~~~text
credit applied: yes/no
project created: yes/no
Compute Engine API enabled: yes/no
A100 quota status: approved/pending/denied
A100 quota value: integer
selected region: asia-northeast1
~~~

Do not share:

~~~text
billing account ID
credit promotion code
payment-card information
service-account private keys
API keys
OAuth tokens
~~~

## Official references

Google Developer Program benefits:
https://developers.google.com/profile/help/benefits

Google AI Pro benefits:
https://support.google.com/googleone/answer/14534406

Cloud Billing budgets:
https://docs.cloud.google.com/billing/docs/how-to/budgets

Compute Engine GPU quotas:
https://docs.cloud.google.com/compute/resource-usage

GPU regions and zones:
https://docs.cloud.google.com/compute/docs/regions-zones/gpu-regions-zones

A2 machine series:
https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines
