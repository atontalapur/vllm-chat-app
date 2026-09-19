-- Trace store: one row per chat request, written after the stream closes.
--
-- Read by the curation query (S3-1), the judge (S2-2), and the failure-rate
-- panel (S5-5). Written only by the api's async writer (S1-3). Never read by
-- the eval harness: pipeline/eval is held out from everything in this table.
--
-- Everything connects as the POSTGRES_USER superuser for now. S1-3 gives the
-- api an INSERT-only role; readers get a SELECT-only one when they land.

CREATE TABLE traces (
    -- X-Request-ID from the api, the join key to the structured log line.
    request_id       text PRIMARY KEY,
    created_at       timestamptz NOT NULL DEFAULT now(),

    -- Model name as served, so base and adapter traces are told apart. This
    -- is the lora_name once S5-3 lands, not the adapter path.
    model            text NOT NULL,

    -- Full conversation as sent upstream (OpenAI messages array) and the
    -- assistant text as accumulated from the stream.
    messages         jsonb NOT NULL,
    response         text NOT NULL,

    -- Confidence proxy from docs/spikes/s0-3-logprob-shape.md. Null when the
    -- logprobs request flag is off. The mean hides one bad token in a long
    -- fluent answer; the minimum does not, so both are kept.
    mean_logprob     double precision,
    min_logprob      double precision,
    n_tokens         integer,

    -- "stop" or "length" from the final chunk. S3-4 drops "length": a
    -- response cut by max_tokens is not a training example.
    finish_reason    text,

    -- Filled in later by the judge and the curation pipeline. Null means
    -- not yet processed, which is how the curation query finds new work.
    judge_score      double precision,
    curation_status  text
        CHECK (curation_status IN ('selected', 'rejected_duplicate',
                                   'rejected_contaminated', 'rejected_quality',
                                   'exported'))
);

-- Curation and the dashboard both scan by time window.
CREATE INDEX traces_created_at_idx ON traces (created_at);

-- The curation query's working set: rows nothing has looked at yet.
CREATE INDEX traces_unprocessed_idx ON traces (created_at)
    WHERE curation_status IS NULL;
