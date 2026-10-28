package com.acme.payments.service;

import java.util.List;
import java.util.Optional;

import org.springframework.stereotype.Service;

import com.acme.payments.domain.Payment;
import com.acme.payments.domain.PaymentResult;
import com.acme.payments.ledger.LedgerPoster;
import com.acme.payments.validation.PaymentRule;

/**
 * The single entry point for every payment, from REST or Kafka.
 * Runs all PaymentRules, then posts accepted payments to the ledger.
 */
@Service
public class PaymentService {

    private final List<PaymentRule> rules;
    private final LedgerPoster ledgerPoster;

    public PaymentService(List<PaymentRule> rules, LedgerPoster ledgerPoster) {
        this.rules = List.copyOf(rules);
        this.ledgerPoster = ledgerPoster;
    }

    public PaymentResult submit(Payment payment) {
        List<String> reasons = rules.stream()
                .map(rule -> rule.check(payment))
                .flatMap(Optional::stream)
                .toList();
        if (!reasons.isEmpty()) {
            return PaymentResult.rejected(payment.id(), reasons);
        }
        ledgerPoster.post(payment);
        return PaymentResult.accepted(payment.id());
    }
}
