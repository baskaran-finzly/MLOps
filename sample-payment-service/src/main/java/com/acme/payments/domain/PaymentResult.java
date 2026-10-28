package com.acme.payments.domain;

import java.util.List;

public record PaymentResult(String paymentId, PaymentStatus status, List<String> reasons) {

    public static PaymentResult accepted(String paymentId) {
        return new PaymentResult(paymentId, PaymentStatus.ACCEPTED, List.of());
    }

    public static PaymentResult rejected(String paymentId, List<String> reasons) {
        return new PaymentResult(paymentId, PaymentStatus.REJECTED, List.copyOf(reasons));
    }
}
