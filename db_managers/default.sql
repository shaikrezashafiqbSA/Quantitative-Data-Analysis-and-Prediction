-- Table: signal_trading.signals
​
-- DROP TABLE signal_trading.signals;
​
CREATE TABLE signal_trading.signals
(
    signal_id integer NOT NULL DEFAULT nextval('signal_trading.signals_signal_id_seq'::regclass),
    model_name character varying(30) COLLATE pg_catalog."default" NOT NULL,
    trade_currency character varying(5) COLLATE pg_catalog."default" NOT NULL,
    trade_type character varying(30) COLLATE pg_catalog."default" NOT NULL,
    quantity numeric NOT NULL,
    pair character varying(10) COLLATE pg_catalog."default" NOT NULL,
    price numeric NOT NULL,
    created_at timestamp with time zone DEFAULT now(),
    updated_at timestamp with time zone DEFAULT now(),
    CONSTRAINT signals_pkey PRIMARY KEY (signal_id)
)
​
TABLESPACE pg_default;
​
ALTER TABLE signal_trading.signals
    OWNER to oms;