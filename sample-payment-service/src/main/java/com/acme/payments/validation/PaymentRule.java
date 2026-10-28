package com.acme.payments.validation;

import java.util.Optional;

import com.acme.payments.domain.Payment;

/**
 * One business rule a payment must pass before it reaches the ledger.
 * Every Spring bean implementing this interface is run by PaymentService,
 * so a rule here applies to REST and Kafka payments alike.
 */
public interface PaymentRule {

    /** Returns a rejection reason, or empty if the payment passes. */
    Optional<String> check(Payment payment);
}
