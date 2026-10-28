package com.acme.payments.validation;

import static org.assertj.core.api.Assertions.assertThat;

import java.math.BigDecimal;
import java.util.Set;

import org.junit.jupiter.api.Test;

import com.acme.payments.config.PaymentProperties;
import com.acme.payments.domain.Payment;
import com.acme.payments.domain.PaymentType;

class PaymentRulesTest {

    @Test
    void positiveAmountRule() {
        PositiveAmountRule rule = new PositiveAmountRule();

        assertThat(rule.check(payment("0.01", "USD", "111", "222"))).isEmpty();
        assertThat(rule.check(payment("0.00", "USD", "111", "222"))).contains("Amount must be greater than zero");
        assertThat(rule.check(payment("-1.00", "USD", "111", "222"))).isPresent();
    }

    @Test
    void supportedCurrencyRule() {
        SupportedCurrencyRule rule = new SupportedCurrencyRule(new PaymentProperties(Set.of("USD"), null));

        assertThat(rule.check(payment("10.00", "USD", "111", "222"))).isEmpty();
        assertThat(rule.check(payment("10.00", "GBP", "111", "222"))).contains("Currency GBP is not supported");
    }

    @Test
    void distinctAccountsRule() {
        DistinctAccountsRule rule = new DistinctAccountsRule();

        assertThat(rule.check(payment("10.00", "USD", "111", "222"))).isEmpty();
        assertThat(rule.check(payment("10.00", "USD", "111", "111"))).contains("Debtor and creditor accounts must be different");
    }

    private static Payment payment(String amount, String currency, String debtor, String creditor) {
        return new Payment("p-1", PaymentType.ACH, new BigDecimal(amount), currency, debtor, creditor);
    }
}
