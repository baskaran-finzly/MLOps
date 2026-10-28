package com.acme.payments.validation;

import java.util.Optional;

import org.springframework.stereotype.Component;

import com.acme.payments.config.PaymentProperties;
import com.acme.payments.domain.Payment;

@Component
public class SupportedCurrencyRule implements PaymentRule {

    private final PaymentProperties properties;

    public SupportedCurrencyRule(PaymentProperties properties) {
        this.properties = properties;
    }

    @Override
    public Optional<String> check(Payment payment) {
        if (!properties.supportedCurrencies().contains(payment.currency())) {
            return Optional.of("Currency " + payment.currency() + " is not supported");
        }
        return Optional.empty();
    }
}
