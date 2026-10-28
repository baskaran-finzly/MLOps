package com.acme.payments.ledger;

import java.math.BigDecimal;
import java.time.Instant;

public record LedgerEntry(String paymentId, String debitAccount, String creditAccount,
                          BigDecimal amount, String currency, Instant postedAt) {
}
