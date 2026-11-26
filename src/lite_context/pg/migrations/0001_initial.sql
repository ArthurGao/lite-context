-- lite-context's transcripts (design §8). Runs with search_path = <schema>.

CREATE TABLE transcripts (
    id          text PRIMARY KEY,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- One row per entry: `data` is a message record (lite_context.codec) or a ContextState. `json`,
-- not `jsonb`: jsonb reorders object keys, and the adapters give vendor blocks back byte for byte
-- (a reordered block misses a prompt-cache prefix).
-- Append-only: the triggers below refuse every UPDATE, DELETE and TRUNCATE, so a stored state's
-- cuts always point at the messages they were measured on.
CREATE TABLE entries (
    transcript_id  text NOT NULL REFERENCES transcripts (id),
    seq            integer NOT NULL CHECK (seq >= 1),
    kind           text NOT NULL CHECK (kind IN ('message', 'state')),
    data           json NOT NULL,
    PRIMARY KEY (transcript_id, seq)
);

CREATE FUNCTION entries_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'lite-context transcript entries are append-only: % refused', TG_OP;
END
$$;

CREATE TRIGGER entries_no_update_or_delete BEFORE UPDATE OR DELETE ON entries
    FOR EACH ROW EXECUTE FUNCTION entries_append_only();
CREATE TRIGGER entries_no_truncate BEFORE TRUNCATE ON entries
    FOR EACH STATEMENT EXECUTE FUNCTION entries_append_only();
