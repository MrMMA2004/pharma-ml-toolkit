import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync, writeFileSync } from 'node:fs';

async function main() {
  const shots: Array<[string, string]> = [
    [
      'pg152-ml-correlations.png',
      'A desktop ML toolkit screenshot (the "Data & ML" tab, the Results card). QA checklist, answer PASS/FAIL per item with one short reason each:\n' +
        '1) The Results card shows a horizontal signed-correlations bar chart with about 10 bars, some teal/green (positive) and at least one in a different warm/red colour (negative), with feature names as y-axis labels.\n' +
        '2) The chart title mentions "Top signed correlations" and "held-out rows", and the x-axis is labelled "Pearson r".\n' +
        '3) The metrics block above the figure shows labels "Pearson r", "train R²" and "R² gap (train−test)" each with a numeric value (the gap signed with + or -).\n' +
        '4) A button row includes buttons reading "Correlations…", "Correlation heatmap" and "Decode ECFP bit…" (possibly with ellipses), all fully visible.\n' +
        '5) No overlapping, clipped, or garbled text anywhere in the window.\n',
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
    const out = name.replace('pg152-', 'vlm152_').replace('.png', '.json');
    writeFileSync(`scratch/${out}`, JSON.stringify(response, null, 2));
    console.log(`${name}: ${(response as any).choices[0].message.content.split('\n').join(' | ')}`);
  }
}

main().catch((error) => {
  console.error('VLM review failed:', error);
  process.exit(1);
});
