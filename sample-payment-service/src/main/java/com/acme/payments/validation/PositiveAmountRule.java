package com.acme.payments.validation;

import java.math.BigDecimal;
import java.util.Optional;

import org.springframework.stereotype.Component;

import com.acme.payments.domain.Payment;

@Component
public class PositiveAmountRule implements PaymentRule {

    @Override
    public Optional<String> check(Payment payment) {
        if (payment.amount() == null || payment.amount().compareTo(BigDecimal.ZERO) <= 0) {
            return Optional.of("Amount must be greater than zero");
        }
        return Optional.empty();
    }
}
