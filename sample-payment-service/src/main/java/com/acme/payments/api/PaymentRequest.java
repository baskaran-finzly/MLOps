package com.acme.payments.api;

import java.math.BigDecimal;

import com.acme.payments.domain.Payment;
import com.acme.payments.domain.PaymentType;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

/** JSON body for POST /payments. Format checks only; business rules live in validation/. */
public record PaymentRequest(
        @NotBlank String id,
        @NotNull PaymentType type,
        @NotNull BigDecimal amount,
        @NotBlank String currency,
        @NotBlank String debtorAccount,
        @NotBlank String creditorAccount) {

    public Payment toPayment() {
        return new Payment(id, type, amount, currency, debtorAccount, creditorAccount);
    }
}
