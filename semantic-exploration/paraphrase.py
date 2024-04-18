import pandas as pd
from openai import OpenAI

client = OpenAI()

test_df = pd.read_json("test.jsonl", lines=True)

paraphrased = []

for i, row in test_df.iterrows():
    before = row["string"]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant to help paraphrase sentences from scientific citations.",
            },
            {
                "role": "user",
                "content": "Paraphrase this scientific sentence for me:" + before,
            },
        ],
    )

    after = completion.choices[0].message.content
    paraphrased.append(after)
    print(i)

df = pd.DataFrame({"string": paraphrased, "label": test_df["label"]})
df.to_json(
    "paraphrased.jsonl",
    lines=True,
    orient="records",
)
