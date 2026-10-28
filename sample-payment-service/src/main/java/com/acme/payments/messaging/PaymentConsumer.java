package com.acme.payments.messaging;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Component;

import com.acme.payments.api.PaymentRequest;
import com.acme.payments.domain.PaymentResult;
import com.acme.payments.service.PaymentService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Consumes payments from Kafka. It goes through PaymentService, exactly like
 * the REST API, so validation rules apply to both channels.
 * Disabled unless payments.kafka.enabled=true.
 */
@Component
@ConditionalOnProperty(name = "payments.kafka.enabled", havingValue = "true")
public class PaymentConsumer {

    private static final Logger log = LoggerFactory.getLogger(PaymentConsumer.class);

    private final PaymentService paymentService;
    private final ObjectMapper objectMapper;

    public PaymentConsumer(PaymentService paymentService, ObjectMapper objectMapper) {
        this.paymentService = paymentService;
        this.objectMapper = objectMapper;
    }

    @KafkaListener(topics = "${payments.kafka.topic}", groupId = "sample-payment-service")
    public PaymentResult onMessage(String json) throws JsonProcessingException {
        PaymentRequest request = objectMapper.readValue(json, PaymentRequest.class);
        PaymentResult result = paymentService.submit(request.toPayment());
        log.info("Payment {} from Kafka: {}", result.paymentId(), result.status());
        return result;
    }
}
