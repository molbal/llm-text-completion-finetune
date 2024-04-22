import os
import argparse
import pyarrow.parquet as pq
import pyarrow as pa
from nltk.tokenize import sent_tokenize
import nltk
import pandas as pd

nltk.download('punkt')


def split_to_chunks(text, split_chunk_size, min_chunk_size):
    sentences = sent_tokenize(text)
    if not sentences: return []
    chunk = [sentences[0]]
    output_chunks = []

    for sentence in sentences[1:]:
        if len(' '.join(chunk)) + len(sentence) <= split_chunk_size:
            chunk.append(sentence)
        else:
            if len(' '.join(chunk)) >= min_chunk_size:
                output_chunks.append(' '.join(chunk))
            chunk = [sentence]
    if chunk:
        output_chunks.append(' '.join(chunk))
    return output_chunks


def main():
    parser = argparse.ArgumentParser(description='Process text files and create a Parquet file.')
    parser.add_argument('--source_dir', type=str, required=True, help='Path to the directory containing text files')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output Parquet file')
    parser.add_argument('--chunk_min', type=str, default=6000, required=False, help='Min chunk size')
    parser.add_argument('--chunk_target', type=str, default=7000, required=False, help='Target chunk size')
    args = parser.parse_args()

    print(f"Input dir: {args.source_dir}")
    print(f"Output file: {args.output_file}")
    print(f"Chunk min: {args.chunk_min}")
    print(f"Chunk target: {args.chunk_target}")

    df_list = []

    for filename in os.listdir(args.source_dir):
        if filename.endswith('.txt'):
            print("Processing file: " + filename)
            filepath = os.path.join(args.source_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    file_contents = f.read()
            except Exception as e:
                print(f"Error occurred: {e}")
                continue
            chunks = split_to_chunks(file_contents, args.chunk_target, args.chunk_min)
            source = filename.replace('.txt', '')

            output_data = [{'chunk': chunk, 'source': source} for chunk in chunks]
            df_list.append(pd.DataFrame(output_data))

            print(f"Appended chunks: {str(len(chunks))} to output data")

    # Write all chunks to parquet
    print(f"Total chunks: {len(df_list)}, writing to output file: {args.output_file}")
    output_df = pd.concat(df_list, ignore_index=True)
    output_df.to_parquet(args.output_file, engine='pyarrow')
    print(f"✅ Done.")


if __name__ == "__main__":
    main()
