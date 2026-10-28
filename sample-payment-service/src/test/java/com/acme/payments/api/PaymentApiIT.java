package com.acme.payments.api;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

/** HTTP-level tests. Runs in `./mvnw verify`, not in `./mvnw test`. */
@SpringBootTest
@AutoConfigureMockMvc
class PaymentApiIT {

    @Autowired
    MockMvc mvc;

    @Test
    void validAchPaymentIsAccepted() throws Exception {
        submit(json("api-1", "ACH", "125.50", "USD"))
                .andExpect(status().isAccepted())
                .andExpect(jsonPath("$.status").value("ACCEPTED"));
    }

    @Test
    void unsupportedCurrencyIsRejectedWithReason() throws Exception {
        submit(json("api-2", "ACH", "125.50", "JPY"))
                .andExpect(status().isUnprocessableEntity())
                .andExpect(jsonPath("$.reasons[0]").value("Currency JPY is not supported"));
    }

    @Test
    void missingAmountIsABadRequest() throws Exception {
        submit("""
                {"id":"api-3","type":"ACH","currency":"USD","debtorAccount":"111","creditorAccount":"222"}
                """)
                .andExpect(status().isBadRequest());
    }

    @Test
    void largeAchPaymentIsAccepted() throws Exception {
        submit(json("api-4", "ACH", "30000.00", "USD"))
                .andExpect(status().isAccepted());
    }

    private org.springframework.test.web.servlet.ResultActions submit(String body) throws Exception {
        return mvc.perform(post("/payments").contentType(MediaType.APPLICATION_JSON).content(body));
    }

    private static String json(String id, String type, String amount, String currency) {
        return """
                {"id":"%s","type":"%s","amount":%s,"currency":"%s","debtorAccount":"111","creditorAccount":"222"}
                """.formatted(id, type, amount, currency);
    }
}
