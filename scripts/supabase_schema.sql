-- atrgrid state 表 —— 整個 State（positions/lots/trades/cash）存成一列 JSONB。
-- 對應 src/atrgrid/state.py 的 State.to_dict()/from_dict()，不拆表，
-- 序列化邏輯完全留在 Python 端，Postgres 只負責原子讀寫這一列。
--
-- 在 Supabase SQL Editor 貼上執行一次即可。

create table if not exists atrgrid_state (
    id smallint primary key default 1,
    data jsonb not null,
    updated_at timestamptz not null default now(),
    constraint atrgrid_state_singleton check (id = 1)
);
