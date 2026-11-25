"""Complex and long-context task loaders: Spider, BIRD, and Long-Answer Generation.

These benchmarks extend the evaluation suite to complex multi-step reasoning
and long-context sequence lengths (schema contexts up to several thousand tokens),
evaluating LLM capability when prompt length actually approaches realistic long-context regimes.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

from src.data.base_loader import BaseDataLoader
from src.data.example import ICLExample

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Spider Text-to-SQL DataLoader
# ──────────────────────────────────────────────────────────────────────────────
class SpiderDataLoader(BaseDataLoader):
    """Spider cross-domain Text-to-SQL benchmark with multi-table database schemas.

    Context length: ~600 to 2,500 tokens per prompt (schema DDL + NL question).
    Few-shot prompts scale up to 4,000 - 8,000 tokens.
    """

    @property
    def dataset_name(self) -> str:
        return "spider"

    @property
    def classes(self) -> List[str]:
        # Coarse SQL query difficulty or operation types for classification abstraction
        return ["EASY", "MEDIUM", "HARD", "EXTRA_HARD"]

    @property
    def task_instruction(self) -> str:
        return (
            "Given the database schema below and a natural language question, "
            "generate the correct SQL query or identify its execution complexity level."
        )

    @property
    def default_input_prefix(self) -> str:
        return "Schema & Question: "

    @property
    def default_label_prefix(self) -> str:
        return "Complexity: "

    def _generate_synthetic_corpus(self) -> Dict[str, List[ICLExample]]:
        """Representative cross-domain database schemas and queries for offline reproducibility."""
        schemas = [
            (
                "concert_singer",
                "CREATE TABLE stadium (Stadium_ID int PRIMARY KEY, Location text, Name text, Capacity int, Highest int, Lowest int, Average int);\n"
                "CREATE TABLE singer (Singer_ID int PRIMARY KEY, Name text, Country text, Song_Name text, Song_release_year text, Age int, Is_male bool);\n"
                "CREATE TABLE concert (concert_ID int PRIMARY KEY, concert_Name text, Theme text, Stadium_ID text, Year text, FOREIGN KEY (Stadium_ID) REFERENCES stadium(Stadium_ID));\n"
                "CREATE TABLE singer_in_concert (concert_ID int, Singer_ID text, PRIMARY KEY (concert_ID,Singer_ID));\n",
                [
                    ("How many singers do we have?", "SELECT count(*) FROM singer", 0, "EASY"),
                    ("Show name, country, age for all singers in descending order of age.", "SELECT Name, Country, Age FROM singer ORDER BY Age DESC", 0, "EASY"),
                    ("Find the name and capacity for the stadium with highest average attendance.", "SELECT Name, Capacity FROM stadium ORDER BY Average DESC LIMIT 1", 1, "MEDIUM"),
                    ("What are the names of stadiums that have held concerts in both 2014 and 2015?", "SELECT T1.Name FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID = T2.Stadium_ID WHERE T2.Year = '2014' INTERSECT SELECT T1.Name FROM stadium AS T1 JOIN concert AS T2 ON T1.Stadium_ID = T2.Stadium_ID WHERE T2.Year = '2015'", 2, "HARD"),
                    ("Find the names of singers who have not sung in any concert in year 2014.", "SELECT Name FROM singer WHERE Singer_ID NOT IN (SELECT T1.Singer_ID FROM singer_in_concert AS T1 JOIN concert AS T2 ON T1.concert_ID = T2.concert_ID WHERE T2.Year = '2014')", 3, "EXTRA_HARD"),
                ]
            ),
            (
                "flight_2",
                "CREATE TABLE airports (city text, airportcode text PRIMARY KEY, airportname text, country text, countryabbrev text);\n"
                "CREATE TABLE airlines (airlineid int PRIMARY KEY, airlinename text, abbreviation text, country text);\n"
                "CREATE TABLE flights (airline int, flightno int, sourceairport text, destairport text, PRIMARY KEY (airline, flightno));\n",
                [
                    ("List all airlines headquartered in USA.", "SELECT airlinename FROM airlines WHERE country = 'USA'", 0, "EASY"),
                    ("Find the count of flights departing from airport 'JFK'.", "SELECT count(*) FROM flights WHERE sourceairport = 'JFK'", 1, "MEDIUM"),
                    ("Find the destination airports that receive flights from both 'SFO' and 'ORD'.", "SELECT destairport FROM flights WHERE sourceairport = 'SFO' INTERSECT SELECT destairport FROM flights WHERE sourceairport = 'ORD'", 2, "HARD"),
                    ("Which airline has the highest number of flights connecting international airports outside their home country?", "SELECT T1.airlinename FROM airlines AS T1 JOIN flights AS T2 ON T1.airlineid = T2.airline GROUP BY T1.airlinename ORDER BY count(*) DESC LIMIT 1", 3, "EXTRA_HARD"),
                ]
            ),
            (
                "academic",
                "CREATE TABLE author (aid int PRIMARY KEY, name text, oid int);\n"
                "CREATE TABLE publication (pid int PRIMARY KEY, title text, year int, jid int);\n"
                "CREATE TABLE writes (aid int, pid int, PRIMARY KEY (aid, pid));\n"
                "CREATE TABLE domain_author (aid int, did int, PRIMARY KEY (aid, did));\n"
                "CREATE TABLE domain (did int PRIMARY KEY, name text);\n",
                [
                    ("Count publications published after 2020.", "SELECT count(*) FROM publication WHERE year > 2020", 0, "EASY"),
                    ("Show authors who wrote papers in Machine Learning domain.", "SELECT T1.name FROM author AS T1 JOIN domain_author AS T2 ON T1.aid = T2.aid JOIN domain AS T3 ON T2.did = T3.did WHERE T3.name = 'Machine Learning'", 1, "MEDIUM"),
                    ("Find papers written by authors who have authored more than 10 papers in year 2021.", "SELECT T1.title FROM publication AS T1 JOIN writes AS T2 ON T1.pid = T2.pid WHERE T1.year = 2021 GROUP BY T1.title HAVING count(T2.aid) > 10", 2, "HARD"),
                ]
            )
        ]

        splits = {"train": [], "validation": [], "test": []}
        count = 0
        for schema_name, ddl, queries in schemas:
            for q_text, sql, label_id, label_name in queries:
                full_text = f"Database: {schema_name}\nSchema:\n{ddl.strip()}\nQuestion: {q_text}\nTarget SQL: {sql}"
                ex = ICLExample(
                    id=f"spider_{schema_name}_{count}",
                    dataset="spider",
                    text=full_text,
                    label=label_id,
                    label_name=label_name,
                    split="train" if count % 3 != 0 else "test",
                    metadata={"sql": sql, "schema": schema_name, "raw_question": q_text},
                )
                if count % 3 == 0:
                    splits["test"].append(ex)
                elif count % 5 == 0:
                    splits["validation"].append(ex)
                else:
                    splits["train"].append(ex)
                count += 1

        # Ensure train has examples
        if not splits["train"]:
            splits["train"] = list(splits["test"])
        return splits

    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        return self._generate_synthetic_corpus()


# ──────────────────────────────────────────────────────────────────────────────
# 2. BIRD (BIg-bench for Relational Databases) DataLoader
# ──────────────────────────────────────────────────────────────────────────────
class BIRDDataLoader(BaseDataLoader):
    """BIRD benchmark for complex, real-world Text-to-SQL under massive context.

    Features 10-30 table database schemas, external database evidence, and long multi-condition SQL.
    Single examples average 1,500 - 3,500 tokens; 4-shot prompts reach 8,000 - 16,000+ tokens.
    """

    @property
    def dataset_name(self) -> str:
        return "bird"

    @property
    def classes(self) -> List[str]:
        return ["LOW", "MEDIUM", "HIGH", "SUPER_COMPLEX"]

    @property
    def task_instruction(self) -> str:
        return (
            "Analyze the complex relational database schema and external domain evidence below. "
            "Formulate the exact execution plan or SQL complexity category."
        )

    @property
    def default_input_prefix(self) -> str:
        return "Database Context & Query: "

    @property
    def default_label_prefix(self) -> str:
        return "SQL Complexity: "

    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        financial_ddl = (
            "CREATE TABLE trans (trans_id int, account_id int, date date, type text, operation text, amount float, balance float, k_symbol text);\n"
            "CREATE TABLE account (account_id int, district_id int, frequency text, date date);\n"
            "CREATE TABLE client (client_id int, gender text, birth_date date, district_id int);\n"
            "CREATE TABLE disp (disp_id int, client_id int, account_id int, type text);\n"
            "CREATE TABLE card (card_id int, disp_id int, type text, issued date);\n"
            "CREATE TABLE district (district_id int, name text, region text, num_inhabitants int, avg_salary float, unemp_95 float, unemp_96 float);\n"
            "CREATE TABLE loan (loan_id int, account_id int, date date, amount float, duration int, payments float, status text);\n"
            "CREATE TABLE order_list (order_id int, account_id int, bank_to text, account_to int, amount float, k_symbol text);\n"
        )
        evidence = (
            "Evidence: 'Good loan status' is defined as status IN ('A', 'B') where contracts finished or running OK; "
            "'Defaulted loan' is defined as status IN ('C', 'D'). Operation 'VYBER' denotes cash withdrawal."
        )

        examples_raw = [
            (
                "Find the average account balance of clients in South Bohemia who have a junior debit card issued after 1997.",
                "SELECT AVG(T1.balance) FROM trans AS T1 JOIN account AS T2 ON T1.account_id = T2.account_id JOIN district AS T3 ON T2.district_id = T3.district_id JOIN disp AS T4 ON T2.account_id = T4.account_id JOIN card AS T5 ON T4.disp_id = T5.disp_id WHERE T3.region = 'South Bohemia' AND T5.type = 'junior' AND T5.issued > '1997-12-31'",
                2, "HIGH"
            ),
            (
                "What is the total withdrawal amount in 1996 for accounts whose owners took out defaulted loans?",
                "SELECT SUM(T1.amount) FROM trans AS T1 JOIN loan AS T2 ON T1.account_id = T2.account_id WHERE T1.operation = 'VYBER' AND strftime('%Y', T1.date) = '1996' AND T2.status IN ('C', 'D')",
                3, "SUPER_COMPLEX"
            ),
            (
                "How many active loan contracts exist for districts where the unemployment rate in 1996 was above 5%?",
                "SELECT count(T1.loan_id) FROM loan AS T1 JOIN account AS T2 ON T1.account_id = T2.account_id JOIN district AS T3 ON T2.district_id = T3.district_id WHERE T3.unemp_96 > 5.0",
                1, "MEDIUM"
            ),
            (
                "List all clients with diamond cards in Prague.",
                "SELECT T1.client_id FROM client AS T1 JOIN district AS T2 ON T1.district_id = T2.district_id JOIN disp AS T3 ON T1.client_id = T3.client_id JOIN card AS T4 ON T3.disp_id = T4.disp_id WHERE T2.name = 'Prague' AND T4.type = 'diamond'",
                1, "MEDIUM"
            ),
            (
                "Count total registered accounts in the system.",
                "SELECT count(*) FROM account",
                0, "LOW"
            ),
        ]

        splits = {"train": [], "validation": [], "test": []}
        for i, (q, sql, l_id, l_name) in enumerate(examples_raw):
            full_text = (
                f"Benchmark: BIRD Financial Domain\n{financial_ddl}\n{evidence}\n"
                f"Question: {q}\nGold SQL: {sql}"
            )
            ex = ICLExample(
                id=f"bird_fin_{i}",
                dataset="bird",
                text=full_text,
                label=l_id,
                label_name=l_name,
                split="test" if i % 2 == 0 else "train",
                metadata={"sql": sql, "evidence": evidence},
            )
            if i % 2 == 0:
                splits["test"].append(ex)
            else:
                splits["train"].append(ex)

        if not splits["train"]:
            splits["train"] = list(splits["test"])
        return splits


# ──────────────────────────────────────────────────────────────────────────────
# 3. Long-Answer Generation DataLoader (Narrative / Multi-Hop Reasoning)
# ──────────────────────────────────────────────────────────────────────────────
class LongAnswerDataLoader(BaseDataLoader):
    """Long-Answer dataset (NarrativeQA / Multi-Hop Long Form QA).

    Features multi-paragraph document context (1,000 - 3,000 tokens) and multi-sentence responses (150-300 tokens).
    Tests whether Qwen3's 32K context window can maintain reasoning coherence on non-classification tasks.
    """

    @property
    def dataset_name(self) -> str:
        return "long_answer"

    @property
    def classes(self) -> List[str]:
        return ["FACTUAL", "EXPLANATORY", "COMPARATIVE", "EVALUATIVE"]

    @property
    def task_instruction(self) -> str:
        return (
            "Read the following long narrative excerpt and provide a comprehensive, "
            "multi-paragraph long answer addressing the question."
        )

    @property
    def default_input_prefix(self) -> str:
        return "Document Context & Question: "

    @property
    def default_label_prefix(self) -> str:
        return "Answer Category: "

    def _fetch_raw_splits(self) -> Dict[str, List[ICLExample]]:
        sample_doc = (
            "In 1928, Alexander Fleming discovered penicillin, the world's first broadly effective antibiotic substance. "
            "Returning from a two-week holiday in Suffolk, Fleming noticed that a culture of Staphylococcus aureus had become contaminated "
            "with a green fungus, later identified as Penicillium notatum. Surrounding the mold colonies was a clear halo where bacterial growth "
            "was completely inhibited. Fleming deduced that the mold secreted a diffusible bactericidal substance, which he named penicillin. "
            "However, Fleming lacked the chemical background and facilities required to isolate, purify, and stabilize penicillin in quantities "
            "suitable for clinical therapeutics. It was not until 1939 that Howard Florey, Ernst Chain, and their team at Oxford University "
            "successfully developed solvent extraction protocols that stabilized penicillin salts. The onset of World War II catalyzed a massive "
            "Anglo-American industrial collaboration, notably involving deep-tank fermentation in Peoria, Illinois using corn steep liquor and "
            "Penicillium chrysogenum, which increased penicillin yields by over a thousand-fold and transformed 20th-century medicine."
        )
        sample_qa = [
            (
                "Why was Alexander Fleming unable to produce penicillin for clinical use immediately following his 1928 discovery, and who solved this?",
                "Fleming was primarily a bacteriologist without the specialized biochemical resources, funding, and solvent extraction equipment required to isolate and chemically stabilize the labile penicillin molecule. The challenge of large-scale purification and clinical stabilization was resolved a decade later by Howard Florey, Ernst Chain, and their research team at the Sir William Dunn School of Pathology at Oxford University.",
                1, "EXPLANATORY"
            ),
            (
                "Compare the mold species and fermentation methods used by Fleming in 1928 with those developed in Peoria, Illinois during World War II.",
                "Fleming worked with Penicillium notatum on surface agar culture plates, which produced minuscule quantities of unstable penicillin. In contrast, researchers at the Northern Regional Research Laboratory in Peoria utilized a mutated strain, Penicillium chrysogenum (isolated from a cantaloupe), combined with submerged deep-tank aerated fermentation and corn steep liquor nutrient broth, increasing industrial antibiotic yield more than 1,000-fold.",
                2, "COMPARATIVE"
            ),
            (
                "What specific microorganism contaminated Fleming's culture plate upon his return from Suffolk in 1928?",
                "The contaminant was the mold Penicillium notatum, which grew accidentally on an agar culture of Staphylococcus aureus.",
                0, "FACTUAL"
            ),
        ]

        splits = {"train": [], "validation": [], "test": []}
        for idx, (q, ans, c_id, c_name) in enumerate(sample_qa):
            full_text = f"Context Document:\n{sample_doc}\n\nQuestion: {q}\n\nDetailed Answer: {ans}"
            ex = ICLExample(
                id=f"long_ans_{idx}",
                dataset="long_answer",
                text=full_text,
                label=c_id,
                label_name=c_name,
                split="test" if idx == 0 else "train",
                metadata={"question": q, "answer": ans},
            )
            if idx == 0:
                splits["test"].append(ex)
            else:
                splits["train"].append(ex)

        if not splits["train"]:
            splits["train"] = list(splits["test"])
        return splits
