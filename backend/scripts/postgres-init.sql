-- Runs once, when the PostgreSQL volume is first created.
-- Separate database for the integration tests, so they never touch real data.
CREATE DATABASE truthlens_test;
