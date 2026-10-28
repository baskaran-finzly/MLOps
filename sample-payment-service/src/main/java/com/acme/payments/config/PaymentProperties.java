package com.acme.payments.config;

import java.util.Set;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Settings under the {@code payments} prefix in application.yml.
 * Limits and thresholds belong here, not as numbers in code.
 */
@ConfigurationProperties(prefix = "payments")
public record PaymentProperties(Set<String> supportedCurrencies, Ach ach) {

    public PaymentProperties {
        supportedCurrencies = supportedCurrencies == null ? Set.of("USD") : Set.copyOf(supportedCurrencies);
        ach = ach == null ? new Ach("17:00") : ach;
    }

    /** ACH-specific settings. */
    public record Ach(String cutoffTime) {
    }
}
