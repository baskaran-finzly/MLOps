package com.acme.payments.api;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.acme.payments.domain.PaymentResult;
import com.acme.payments.domain.PaymentStatus;
import com.acme.payments.service.PaymentService;

import jakarta.validation.Valid;

@RestController
@RequestMapping("/payments")
public class PaymentController {

    private final PaymentService paymentService;

    public PaymentController(PaymentService paymentService) {
        this.paymentService = paymentService;
    }

    @PostMapping
    public ResponseEntity<PaymentResult> submit(@Valid @RequestBody PaymentRequest request) {
        PaymentResult result = paymentService.submit(request.toPayment());
        HttpStatus status = result.status() == PaymentStatus.ACCEPTED
                ? HttpStatus.ACCEPTED
                : HttpStatus.UNPROCESSABLE_ENTITY;
        return ResponseEntity.status(status).body(result);
    }
}
