package com.acme.payments.validation;

import java.util.Objects;
import java.util.Optional;

import org.springframework.stereotype.Component;

import com.acme.payments.domain.Payment;

@Component
public class DistinctAccountsRule implements PaymentRule {

    @Override
    public Optional<String> check(Payment payment) {
        if (Objects.equals(payment.debtorAccount(), payment.creditorAccount())) {
            return Optional.of("Debtor and creditor accounts must be different");
        }
        return Optional.empty();
    }
}
