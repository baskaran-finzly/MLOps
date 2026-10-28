package com.acme.payments.ledger;

import java.time.Clock;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import org.springframework.stereotype.Component;

import com.acme.payments.domain.Payment;

/**
 * Posts accepted payments to the ledger.
 *
 * PROTECTED CODE: changes here move real money in production, so they are
 * made by humans and reviewed by the ledger team. Business rules do not
 * belong here; add a PaymentRule in the validation package instead.
 *
 * This sample keeps entries in memory so it runs without a database.
 */
@Component
public class LedgerPoster {

    private final List<LedgerEntry> entries = new CopyOnWriteArrayList<>();
    private final Clock clock;

    public LedgerPoster() {
        this(Clock.systemUTC());
    }

    LedgerPoster(Clock clock) {
        this.clock = clock;
    }

    public LedgerEntry post(Payment payment) {
        LedgerEntry entry = new LedgerEntry(payment.id(), payment.debtorAccount(), payment.creditorAccount(),
                payment.amount(), payment.currency(), clock.instant());
        entries.add(entry);
        return entry;
    }

    public List<LedgerEntry> entries() {
        return List.copyOf(entries);
    }
}
