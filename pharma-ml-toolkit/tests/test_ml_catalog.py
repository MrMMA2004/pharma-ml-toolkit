"""Best-of-ML catalogue regression pins (r368 / part19 Phase 5).

The ``pharmaml.ml.catalog`` snapshot (924 projects across 35 sections,
the honest per-category coverage audit) previously rode only the
module doctests.  These pins make the snapshot's invariants a normal
suite member so a CSV/data edit or a refactor cannot silently drift
the catalogue's headline numbers or the audit's verdict vocabulary.
"""

from __future__ import annotations

from pharmaml.ml import catalog


class TestCatalogSnapshot:
    def test_headline_numbers(self) -> None:
        summary = catalog.catalog_summary()
        assert summary.n_projects == 924
        assert summary.n_categories == 35
        assert summary.total_stars >= 5_000_000
        assert summary.snapshot  # the dated CSV snapshot id is present
        assert summary.biggest_category == "Text Data & NLP"

    def test_categories_and_projects_agree(self) -> None:
        categories = catalog.catalog_categories()
        assert len(categories) == 35
        projects = catalog.catalog_projects()
        assert len(projects) == 924
        # every category's count equals the projects resolved into it
        counts: dict[str, int] = {}
        for project in projects:
            counts[project.category_title] = counts.get(project.category_title, 0) + 1
        for category in categories:
            assert counts.get(category.title, 0) == category.project_count, (
                f"{category.title}: {counts.get(category.title, 0)} rows vs "
                f"{category.project_count} declared"
            )
        # star totals agree between the two views
        assert sum(c.total_stars for c in categories) == sum(p.stars for p in projects)

    def test_projects_are_catalogue_ordered(self) -> None:
        # the catalogue order is category-blocked with score-descending
        # rows inside each category block (search keeps this order too)
        projects = catalog.catalog_projects()
        from itertools import groupby

        for _cat, grp in groupby(projects, key=lambda p: p.category_title):
            scores = [p.score for p in grp]
            assert scores == sorted(scores, reverse=True)
        assert all(p.name and p.category and p.homepage for p in projects)

    def test_search_catalog_is_bounded_and_case_insensitive(self) -> None:
        hits = catalog.search_catalog("TORCH", limit=10)
        assert 0 < len(hits) <= 10
        assert any("pytorch" in p.name.lower() for p in hits)
        assert catalog.search_catalog("zzz-no-such-project-xyz", limit=5) == []


class TestCoverageAudit:
    def test_verdict_vocabulary_and_counts(self) -> None:
        table = catalog.coverage_table()
        assert len(table) == 35
        statuses = {entry.status for entry in table}
        assert statuses <= {
            catalog.COVERAGE_NATIVE,
            catalog.COVERAGE_PARTIAL,
            catalog.COVERAGE_EXTERNAL,
            catalog.COVERAGE_OUT_OF_SCOPE,
        }
        # every verdict class is actually used (the audit stays honest,
        # not everything painted native)
        assert catalog.COVERAGE_NATIVE in statuses
        assert catalog.COVERAGE_EXTERNAL in statuses
        native = [e for e in table if e.status == catalog.COVERAGE_NATIVE]
        assert len(native) >= 12
        # every entry carries its verdict note; the native/partial
        # verdicts must name the pharmaml surfaces they claim, while
        # external/out-of-scope entries may defer with an empty
        # surfaces tuple (the note IS the deferral)
        assert all(entry.note for entry in table)
        for entry in table:
            if entry.status in (catalog.COVERAGE_NATIVE, catalog.COVERAGE_PARTIAL):
                assert entry.surfaces, (
                    f"{entry.slug} verdict {entry.status} must name its surfaces"
                )

    def test_coverage_covers_all_categories(self) -> None:
        table = catalog.coverage_table()
        covered = {entry.slug for entry in table}
        for category in catalog.catalog_categories():
            assert category.slug in covered, f"no coverage verdict for {category.slug}"
