# sample-payment-service

Spring Boot 3 service that accepts payments (ACH, book transfer) over REST
and from the Kafka topic `payments.inbound`, validates them, and posts them
to the ledger.

## Architecture
- `api/`         REST controllers. No business rules here.
- `messaging/`   Kafka consumers. They call PaymentService, same as the API.
- `service/`     PaymentService runs every PaymentRule, then calls the ledger.
- `validation/`  One class per rule, each implementing PaymentRule.
- `ledger/`      Posting to the ledger. PROTECTED: humans change this, not agents.
- `config/`      PaymentProperties, bound from `payments.*` in application.yml.

## Rules
- Money is always BigDecimal with RoundingMode.HALF_EVEN. Never double or float.
- New validation goes in `validation/` as a PaymentRule `@Component`, so REST
  and Kafka payments both get it.
- Limits and thresholds come from application.yml via PaymentProperties,
  never hard-coded numbers.
- Every new rule gets unit tests, including the exact boundary values.
- If an existing test encodes behavior the task changes, update it and say so
  in your summary. Never delete a test to make the build pass.
- Never open `.env` or anything in `secrets/`.
- Never push. Leave commits for a human to review.

## Commands
- Unit tests (fast): `./mvnw -q test`
- Full check, including `*IT.java` integration tests: `./mvnw -q verify`
- Run the tests before saying a task is done.
