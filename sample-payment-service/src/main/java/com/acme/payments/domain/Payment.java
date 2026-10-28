package com.acme.payments.domain;

import java.math.BigDecimal;

/**
 * A payment instruction, whether it arrived over REST or from Kafka.
 * Money is always BigDecimal.
 */
public record Payment(
        String id,
        PaymentType type,
        BigDecimal amount,
        String currency,
        String debtorAccount,
        String creditorAccount) {
}
