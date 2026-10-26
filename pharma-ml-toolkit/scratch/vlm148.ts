import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync } from 'node:fs';

async function main() {
  const shots: Array<[string, string]> = [
    [
      'pg148-ml-pr.png',
      'A desktop ML toolkit screenshot (the "Data & ML" tab). QA checklist, answer PASS/FAIL per item with one short reason each:\n' +
        '1) The Results card shows a Precision-Recall figure with multiple per-class curves, a legend containing "AP" entries, and dashed horizontal no-skill baseline lines.\n' +
        '2) Above the figure, the metrics block shows a "per class" line listing classes (inactive/moderate/strong) with P/R/F1/n values.\n' +
        '3) The Results card has visible buttons including "ROC curves" and "PR curve".\n' +
        '4) No overlapping, clipped, or garbled text anywhere.',
    ],
    [
      'pg148-families.png',
      'A desktop chemistry toolkit screenshot (the "Chemistry" tab, feature factory card). QA checklist, answer PASS/FAIL per item with one short reason each:\n' +
        '1) The family checkbox grid shows twenty feature-family checkboxes including MACCS keys (167), FCFP bits, topological torsion, E-State counts (79), ERG reduced graph (315), and 3D shape descriptors.\n' +
        '2) A width label reads "20 families selected · ~27,759 columns will be computed" (numbers may vary slightly).\n' +
        '3) The checkboxes are aligned in a tidy grid without overlapping text.\n' +
        '4) No clipped or garbled text anywhere.',
    ],
    [
      'pg148-ml-triage.png',
      'A desktop ML toolkit screenshot (the "Data & ML" persistence card). QA checklist, answer PASS/FAIL per item with one short reason each:\n' +
        '1) The "Load a bundle & predict" side shows buttons "pIC50 → IC50 nM" and "Triage hits" plus a potency threshold combobox reading "100" with "nM".\n' +
        '2) The info line under them mentions "3 of 6 row(s) at or below 100 nM" (or similar honest census wording).\n' +
        '3) The prediction table shows fewer rows than a full table would (the triaged hits), with an IC50_nM column.\n' +
        '4) No overlapping, clipped, or garbled text anywhere.',
    ],
    [
      'pg148-ml-benchmark.png',
      'A desktop ML toolkit screenshot (the "Data & ML" Leaderboard card). QA checklist, answer PASS/FAIL per item with one short reason each:\n' +
        '1) The leaderboard table shows about 18 rows, one per feature family (family names like estate, maccs, ecfp, pattern, layered...).\n' +
        '2) The table columns include family, n_features, n_samples, a score mean/std pair, fit_time_s and folds.\n' +
        '3) The status line above the table mentions "2-fold family benchmark over the 18 2D families" and "shape_3d and rism opt-in only".\n' +
        '4) A "Benchmark feature families" button with model and folds inputs is visible.\n' +
        '5) No overlapping, clipped, or garbled text anywhere.',
    ],
  ];
  const zai = await ZAI.create();
  for (const [name, prompt] of shots) {
    const data = readFileSync(`download/${name}`);
    const url = `data:image/png;base64,${data.toString('base64')}`;
    const response = await zai.chat.completions.createVision({
      model: 'glm-5v-turbo',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: prompt },
            { type: 'image_url', image_url: { url } },
          ],
        },
      ],
    });
    const out = name.replace('pg148-', 'vlm148_').replace('.png', '.json');
    writeFileSync(`scratch/${out}`, JSON.stringify(response, null, 2));
    console.log(`${name}: ${(response as any).choices[0].message.content.split('\n').join(' | ')}`);
  }
}

main().catch((error) => {
  console.error('VLM review failed:', error);
  process.exit(1);
});
