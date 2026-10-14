"""The quality suite's own checkers, offline: IFEval instruction checks, the JSON validator, tool-call matching,
the repetition metric, the long-context and soak document generators, the sandbox image and the PNG writer."""

from __future__ import annotations

import json
import struct
import sys
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quality import (docvision, exactness, humaneval, ifeval, json_schema, longtasks, math500, mgsm, mmlu,  # noqa: E402
                     needles, repetition, soak, tools)


class IfevalChecks(unittest.TestCase):
    def test_each_implemented_instruction_has_a_passing_and_a_failing_case(self) -> None:
        cases = {
            "keywords:existence": ({"keywords": ["apple", "pear"]}, "an apple and a pear", "an apple"),
            "keywords:frequency": ({"keyword": "go", "relation": "at least", "frequency": 2}, "go go go", "go"),
            "keywords:forbidden_words": ({"forbidden_words": ["never"]}, "always", "never say"),
            "keywords:letter_frequency": ({"letter": "z", "let_relation": "less than", "let_frequency": 2}, "zero", "zz"),
            "length_constraints:number_words": ({"relation": "at least", "num_words": 3}, "a b c d", "a b"),
            "length_constraints:number_sentences": ({"relation": "less than", "num_sentences": 3}, "One. Two.", "A. B. C."),
            "length_constraints:number_paragraphs": ({"num_paragraphs": 2}, "p1\n***\np2", "p1"),
            "length_constraints:nth_paragraph_first_word": ({"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "hello"}, "a\n\nhello b", "a\n\nbye b"),
            "detectable_content:number_placeholders": ({"num_placeholders": 2}, "[name] at [place]", "[name]"),
            "detectable_content:postscript": ({"postscript_marker": "P.S."}, "text\nP.S. more", "text"),
            "detectable_format:number_bullet_lists": ({"num_bullets": 2}, "* a\n* b", "* a"),
            "detectable_format:constrained_response": ({}, "My answer is no.", "No."),
            "detectable_format:number_highlighted_sections": ({"num_highlights": 1}, "*hi* there", "hi there"),
            "detectable_format:multiple_sections": ({"section_spliter": "SECTION", "num_sections": 2}, "SECTION 1\nx\nSECTION 2\ny", "SECTION 1\nx"),
            "detectable_format:json_format": ({}, '{"a": 1}', "not json"),
            "detectable_format:title": ({}, "<<T>> body", "body"),
            "combination:two_responses": ({}, "a\n******\nb", "a"),
            "combination:repeat_prompt": ({"prompt_to_repeat": "Say hi"}, "Say hi\nhi", "hi"),
            "startend:end_checker": ({"end_phrase": "Is there anything else?"}, "x Is there anything else?", "x"),
            "startend:quotation": ({}, '"quoted"', "bare"),
            "change_case:capital_word_frequency": ({"capital_relation": "at least", "capital_frequency": 1}, "ONE two", "one two"),
            "change_case:english_capital": ({}, "ALL CAPS", "Not caps"),
            "change_case:english_lowercase": ({}, "all lower", "Not lower"),
            "punctuation:no_comma": ({}, "no comma", "a, b"),
        }
        self.assertEqual(set(cases), set(ifeval.IMPLEMENTED))
        for instr, (kw, good, bad) in cases.items():
            self.assertTrue(ifeval.check(instr, kw, good), instr)
            self.assertFalse(ifeval.check(instr, kw, bad), instr)
        self.assertIsNone(ifeval.check("language:response_language", {}, "x"))

    def test_loose_variants_strip_a_leading_line_and_markers(self) -> None:
        v = ifeval.variants("Sure, here it is:\nNO COMMAS HERE")
        self.assertIn("NO COMMAS HERE", v)
        self.assertTrue(any(ifeval.check("change_case:english_capital", {}, x) for x in v))


class JsonValidator(unittest.TestCase):
    def test_types_required_enum_bounds_and_extra_keys(self) -> None:
        schema = json_schema.CASES[2][1]
        self.assertEqual(json_schema.validate({"celsius": 21.5, "hour": 7, "status": "ok"}, schema), [])
        self.assertTrue(json_schema.validate({"celsius": "21", "hour": 7, "status": "ok"}, schema))
        self.assertTrue(json_schema.validate({"celsius": 21.5, "hour": 24, "status": "ok"}, schema))
        self.assertTrue(json_schema.validate({"celsius": 21.5, "hour": 7, "status": "fine"}, schema))
        self.assertTrue(json_schema.validate({"celsius": 21.5, "hour": True, "status": "ok"}, schema))
        strict = json_schema.CASES[0][1]
        self.assertTrue(json_schema.validate({"name": "a", "born": 1815, "city": "L", "extra": 1}, strict))

    def test_extract_json_from_prose_and_fences(self) -> None:
        self.assertEqual(json_schema.extract_json('Here: ```json\n{"a": [1, 2]}\n``` done'), {"a": [1, 2]})
        self.assertEqual(json_schema.extract_json('[1, 2] trailing'), [1, 2])
        with self.assertRaises(ValueError):
            json_schema.extract_json("no json here")

    def test_thirty_cases(self) -> None:
        self.assertEqual(len(json_schema.cases()), 30)


class ToolMatching(unittest.TestCase):
    def test_exact_arguments_and_sets(self) -> None:
        want = [("get_weather", {"city": "Rome"}), ("get_weather", {"city": "Madrid"})]
        self.assertTrue(tools._match([("get_weather", {"city": "Madrid"}), ("get_weather", {"city": "Rome"})], want))
        self.assertFalse(tools._match([("get_weather", {"city": "Rome"})], want))
        self.assertFalse(tools._match([("get_weather", {"city": "Rome", "unit": "celsius"})], [("get_weather", {"city": "Rome"})]))
        self.assertTrue(tools._match([("convert_currency", {"amount": 250.0, "from_currency": "EUR", "to_currency": "USD"})],
                                     [("convert_currency", {"amount": 250, "from_currency": "EUR", "to_currency": "USD"})]))
        self.assertEqual(len(tools.CASES), 60)
        self.assertEqual(sum(1 for _, want in tools.CASES if not want), 9)
        names = {t["function"]["name"] for t in tools.TOOLS}
        for _, want in tools.CASES:
            for name, _ in want:
                self.assertIn(name, names)


class Metrics(unittest.TestCase):
    def test_repetition_rate(self) -> None:
        self.assertEqual(repetition.repeated_4gram_rate("a b c d e f g h"), 0.0)
        self.assertGreater(repetition.repeated_4gram_rate("a b c d " * 20), 0.9)

    def test_documents_are_repeatable_and_sized(self) -> None:
        a, b = needles.document(8000, 3), needles.document(8000, 3)
        self.assertEqual(a, b)
        chars = sum(len(p) for p in a)
        self.assertTrue(6000 * 3.9 < chars < 10000 * 3.9)

    def test_exactness_prompt_set(self) -> None:
        self.assertEqual(len(exactness.PROMPTS), 12)
        self.assertEqual(len(set(exactness.PROMPTS)), 12)


class SoakAndSandbox(unittest.TestCase):
    def test_soak_chats_are_repeatable_distinct_and_long(self) -> None:
        a, b = soak.history(0, 4600), soak.history(0, 4600)
        self.assertEqual(a, b)
        self.assertNotEqual(soak.history(0, 4600)[1]["content"], soak.history(1, 4600)[1]["content"])
        doc, fact = soak.document(2, 4600)
        self.assertIn(fact, doc)
        self.assertGreater(len(doc.split()), 4600)
        self.assertEqual([m["role"] for m in a], ["system", "user", "assistant", "user"])
        later = soak.history(0, 4600, turn=3)
        self.assertEqual(later[:len(a) - 1], a[:-1])            # a later turn extends the earlier ones
        self.assertEqual(len(later), len(a) + 6)

    def test_sandbox_defaults_to_the_recipe_image(self) -> None:
        import tomllib
        with open(Path(__file__).resolve().parents[1] / "recipe.toml", "rb") as f:
            self.assertEqual(humaneval.recipe_image(), tomllib.load(f)["engine"]["image"])

if __name__ == "__main__":
    unittest.main()


class NewChecks(unittest.TestCase):
    def test_mmlu_letter_parsing_and_sample(self) -> None:
        self.assertEqual(mmlu.letter("reasoning...\nAnswer: C"), "C")
        self.assertEqual(mmlu.letter("Answer: (B)"), "B")
        self.assertEqual(mmlu.letter("I think it is D"), "D")
        self.assertIsNone(mmlu.letter("no idea"))
        rows = mmlu.sample(2)
        self.assertEqual(len(rows), 114)
        self.assertEqual(rows, mmlu.sample(2))           # the same questions every run

    def test_mgsm_rows(self) -> None:
        for lang in mgsm.LANGS:
            r = mgsm.rows(lang)
            self.assertEqual(len(r), 250, lang)
            self.assertTrue(r[0][1].lstrip("-").replace(".", "").isdigit(), lang)

    def test_humaneval_extract(self) -> None:
        pr = humaneval.problems()[0]
        body = pr["canonical_solution"]
        full = f"```python\n{pr['prompt']}{body}```"
        self.assertIn(f"def {pr['entry_point']}", humaneval.extract(full, pr["prompt"], pr["entry_point"]))
        bare = humaneval.extract(f"```python\n{body}```", pr["prompt"], pr["entry_point"])
        self.assertTrue(bare.startswith(pr["prompt"]))
        self.assertEqual(len(humaneval.problems()), 164)


@unittest.skipUnless(__import__("shutil").which("docker") and __import__("subprocess").run(
    ["docker", "image", "inspect", "tf-glm53-flash:native-v1.0.5"], capture_output=True).returncode == 0,
    "needs docker and the recipe image")
class HumanEvalSandbox(unittest.TestCase):
    def test_canonical_solutions_pass_and_a_wrong_one_fails(self) -> None:
        probs = humaneval.problems()[:8]
        progs = {p["task_id"].replace("/", "_"): humaneval.program(p, p["prompt"] + p["canonical_solution"]) for p in probs}
        progs["wrong"] = humaneval.program(probs[0], probs[0]["prompt"] + "    return None\n")
        progs["net"] = "import urllib.request\nurllib.request.urlopen('http://1.1.1.1', timeout=3)\n"
        v = humaneval.execute(progs, "tf-glm53-flash:native-v1.0.5")
        for p in probs:
            self.assertEqual(v[p["task_id"].replace("/", "_")], "pass")
        self.assertTrue(v["wrong"].startswith("fail"))
        self.assertTrue(v["net"].startswith("fail"))     # no network inside the sandbox


class ToolEquivalence(unittest.TestCase):
    def test_same_value_or_place_written_differently(self) -> None:
        m = tools._match
        self.assertTrue(m([("calculate", {"expression": "2**20"})], [("calculate", {"expression": "2 ** 20"})]))
        self.assertTrue(m([("calculate", {"expression": "15/100*240"})], [("calculate", {"expression": "0.15 * 240"})]))
        self.assertFalse(m([("calculate", {"expression": "2**21"})], [("calculate", {"expression": "2 ** 20"})]))
        self.assertFalse(m([("calculate", {"expression": "two to the twentieth"})], [("calculate", {"expression": "2 ** 20"})]))
        want = [("get_directions", {"origin": "the Louvre", "destination": "Notre-Dame", "mode": "walking"})]
        self.assertTrue(m([("get_directions", {"origin": "louvre, paris", "destination": "notre-dame cathedral, paris", "mode": "walking"})], want))
        self.assertFalse(m([("get_directions", {"origin": "eiffel tower", "destination": "notre-dame", "mode": "walking"})], want))
        self.assertFalse(m([("get_directions", {"origin": "louvre", "destination": "notre-dame de reims", "mode": "walking"})], want))
        self.assertFalse(m([("get_directions", {"origin": "louvre", "destination": "notre-dame", "mode": "driving"})], want))
        self.assertFalse(m([("get_directions", {"origin": "louvre", "destination": "notre-dame"})], want))   # a dropped argument
        self.assertTrue(m([("get_weather", {"city": "Reykjavik, Iceland"})], [("get_weather", {"city": "Reykjavik"})]))
        self.assertFalse(m([("get_weather", {"city": "Reykjavik", "unit": "celsius"})], [("get_weather", {"city": "Reykjavik"})]))
        # other text arguments stay exact (after case and a trailing full stop)
        self.assertFalse(m([("translate", {"text": "good morning everyone", "target_language": "Spanish"})],
                           [("translate", {"text": "good morning", "target_language": "Spanish"})]))


class ReasoningAndVisionGraders(unittest.TestCase):
    def test_math_boxed_and_equivalent_forms(self) -> None:
        self.assertEqual(math500.boxed("so \\boxed{\\frac{1}{\\sqrt{2}}} and \\boxed{3}"), "3")
        self.assertEqual(math500.boxed("first \\boxed{\\frac{1}{\\sqrt{2}}}"), "\\frac{1}{\\sqrt{2}}")
        self.assertIsNone(math500.boxed("no box here"))
        for a, b in (("\\frac{1}{2}", "0.5"), ("\\dfrac{3}{4}", "\\frac34"), ("x = 5", "5"), ("10^\\circ", "10"),
                     ("1,000", "1000"), ("\\text{(C)}", "(C)"), ("2\\sqrt{3}", "2\\sqrt3"), ("6.0", "6"), ("3/4", "0.75")):
            self.assertTrue(math500.same(a, b), (a, b))
        for a, b in (("\\frac{1}{3}", "0.33"), ("5", "6"), ("(1,2)", "(2,1)")):
            self.assertFalse(math500.same(a, b), (a, b))
        self.assertFalse(math500.same(None, "1"))

    def test_chartqa_relaxed_and_docvqa_anls(self) -> None:
        self.assertTrue(docvision.relaxed("20.5", "21"))
        self.assertFalse(docvision.relaxed("19", "21"))
        self.assertTrue(docvision.relaxed("Yes.", "yes"))
        self.assertTrue(docvision.relaxed("45%", "45"))
        self.assertEqual(docvision.anls("Mid Year", ["mid year"]), 1.0)
        self.assertAlmostEqual(docvision.anls("Mid-Year", ["mid year"]), 0.875)
        self.assertEqual(docvision.anls("completely different", ["0.28"]), 0.0)


class LongTasks(unittest.TestCase):
    class Echo:
        """A fake server: answers every long-task question correctly from the prompt itself."""

        def chat(self, messages, max_tokens=0, **_):
            import re
            p = messages[0]["content"]
            q = p.rsplit("Question: ", 1)[1]
            if "special magic number for " in q and "all" not in q:
                key = re.search(r"number for (\S+) mentioned", q).group(1)
                ans = re.search(rf"numbers for {re.escape(key)} is: (\d+)", p).group(1)
            elif "all the special magic numbers" in q:
                key = re.search(r"numbers for (\S+) mentioned", q).group(1)
                ans = ", ".join(re.findall(rf"numbers for {re.escape(key)} is: (\d+)", p))
            elif "assigned the value" in q:
                v = re.search(r"value (\d+)", q).group(1)
                names = set(re.findall(rf"VAR ([A-Z]{{3}}) = {v}\b", p))
                grew = True
                while grew:
                    new = {a for a, b in re.findall(r"VAR ([A-Z]{3}) = VAR ([A-Z]{3})", p) if b in names}
                    grew = not new <= names
                    names |= new
                ans = ", ".join(sorted(names))
            else:
                from collections import Counter
                ans = ", ".join(w for w, _ in Counter(re.findall(r"\d+\. ([a-z]+)", p)).most_common(10))
            return {"choices": [{"message": {"content": ans}}], "usage": {"prompt_tokens": len(p) // 4}, "_seconds": 0.01}

    def test_a_correct_reader_scores_everything(self) -> None:
        r = longtasks.run(self.Echo(), max_tokens=32_000)
        self.assertEqual((r.passed, r.cases), (20, 20), r.notes)
