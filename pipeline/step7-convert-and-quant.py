import argparse

from unsloth import FastLanguageModel


def main():
    parser = argparse.ArgumentParser(description='Convert and quantize model.')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Path to the checkpoint containing the exported adapter.')
    parser.add_argument('--output_dir', type=str, default="/output-quantized", required=False,
                        help='Target directory to save the quantized model.')
    parser.add_argument('--quantization_method', type=str, default="q4_k_m", required=False,
                        help='Method of quantization.')
    args = parser.parse_args()

    max_seq_length = 8192  # Please change it to match the value in step6-train.py
    dtype = None
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.source_dir,  # "/output/checkpoint-500",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model.save_pretrained_gguf(args.output_dir, tokenizer, quantization_method=args.quantization_method)
    print("Saved model to directory: " + args.output_dir)
    print(f"✅ Done.")


if __name__ == "__main__":
    main()
