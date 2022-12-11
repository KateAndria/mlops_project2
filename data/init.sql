CREATE TABLE public.data
(
    "age"      INT,
    "sex"      INT,
    "cp"       INT,
    "trtbps"   INT,
    "chol"     INT,
    "fbs"      INT,
    "restecg"  INT,
    "thalachh" INT,
    "exng"     INT,
    "oldpeak"  NUMERIC(18, 4),
    "slp"      INT,
    "caa"      INT,
    "thall"    INT,
    "output"   INT
);

COPY public.data(
  "age",
  "sex",
  "cp",
  "trtbps",
  "chol",
  "fbs",
  "restecg",
  "thalachh",
  "exng",
  "oldpeak",
  "slp",
  "caa",
  "thall",
  "output"
) FROM '/var/lib/postgresql/data/heart.csv' DELIMITER ',' CSV HEADER;

CREATE TABLE public.models
(
    "model_id"         TEXT PRIMARY KEY,
    "model_name"       TEXT    NOT NULL,
    "model_params"     TEXT,
    "model_is_trained" BOOLEAN NOT NULL DEFAULT False,
    "model_weights"    BYTEA
);
