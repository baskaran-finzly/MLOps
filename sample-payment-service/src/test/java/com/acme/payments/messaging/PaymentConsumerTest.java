package com.acme.payments.messaging;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

import com.acme.payments.domain.PaymentStatus;
import com.acme.payments.ledger.LedgerPoster;
import com.acme.payments.service.PaymentService;
import com.acme.payments.validation.PositiveAmountRule;
import com.fasterxml.jackson.databind.ObjectMapper;

class PaymentConsumerTest {

    @Test
    void kafkaMessagesGoThroughTheSameRulesAsRest() throws Exception {
        LedgerPoster ledger = new LedgerPoster();
        PaymentService service = new PaymentService(List.of(new PositiveAmountRule()), ledger);
        PaymentConsumer consumer = new PaymentConsumer(service, new ObjectMapper());

        var accepted = consumer.onMessage("""
                {"id":"k-1","type":"ACH","amount":99.95,"currency":"USD","debtorAccount":"111","creditorAccount":"222"}
                """);
        var rejected = consumer.onMessage("""
                {"id":"k-2","type":"ACH","amount":0,"currency":"USD","debtorAccount":"111","creditorAccount":"222"}
                """);

        assertThat(accepted.status()).isEqualTo(PaymentStatus.ACCEPTED);
        assertThat(rejected.status()).isEqualTo(PaymentStatus.REJECTED);
        assertThat(ledger.entries()).extracting("paymentId").containsExactly("k-1");
    }
}
