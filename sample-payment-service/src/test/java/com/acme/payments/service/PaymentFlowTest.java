package com.acme.payments.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.math.BigDecimal;
import java.util.UUID;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import com.acme.payments.domain.Payment;
import com.acme.payments.domain.PaymentResult;
import com.acme.payments.domain.PaymentStatus;
import com.acme.payments.domain.PaymentType;
import com.acme.payments.ledger.LedgerPoster;

/** End-to-end flow through every registered PaymentRule and the ledger, without HTTP or Kafka. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.NONE)
class PaymentFlowTest {

    @Autowired
    PaymentService paymentService;

    @Autowired
    LedgerPoster ledgerPoster;

    @Test
    void acceptedPaymentIsPostedToLedger() {
        Payment payment = ach("1250.00");

        PaymentResult result = paymentService.submit(payment);

        assertThat(result.status()).isEqualTo(PaymentStatus.ACCEPTED);
        assertThat(ledgerPoster.entries()).anyMatch(e -> e.paymentId().equals(payment.id()));
    }

    @Test
    void rejectedPaymentIsNotPostedToLedger() {
        Payment payment = new Payment(newId(), PaymentType.ACH, new BigDecimal("10.00"), "EUR", "111", "222");

        PaymentResult result = paymentService.submit(payment);

        assertThat(result.status()).isEqualTo(PaymentStatus.REJECTED);
        assertThat(result.reasons()).containsExactly("Currency EUR is not supported");
        assertThat(ledgerPoster.entries()).noneMatch(e -> e.paymentId().equals(payment.id()));
    }

    @Test
    void allFailingRulesAreReported() {
        Payment payment = new Payment(newId(), PaymentType.BOOK_TRANSFER, new BigDecimal("-5.00"), "USD", "111", "111");

        PaymentResult result = paymentService.submit(payment);

        assertThat(result.reasons()).containsExactlyInAnyOrder(
                "Amount must be greater than zero",
                "Debtor and creditor accounts must be different");
    }

    @Test
    void largeAchPaymentAccepted() {
        PaymentResult result = paymentService.submit(ach("30000.00"));

        assertThat(result.status()).isEqualTo(PaymentStatus.ACCEPTED);
    }

    @Test
    void largeBookTransferAccepted() {
        Payment payment = new Payment(newId(), PaymentType.BOOK_TRANSFER, new BigDecimal("250000.00"), "USD", "111", "222");

        assertThat(paymentService.submit(payment).status()).isEqualTo(PaymentStatus.ACCEPTED);
    }

    private static Payment ach(String amount) {
        return new Payment(newId(), PaymentType.ACH, new BigDecimal(amount), "USD", "111", "222");
    }

    private static String newId() {
        return UUID.randomUUID().toString();
    }
}
