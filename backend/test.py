from onnx_genai_runner import ONNXGenAIRunner

def main():
    # 1. Point to your model path
    model_path = r"C:\models\phi_models\models--microsoft--Phi-3.5-mini-instruct-onnx\snapshots\aded733f0b665ac2e21ffd8a008f82eb4278a134\cpu_and_mobile\cpu-int4-awq-block-128-acc-level-4"

    # 2. Create runner
    runner = ONNXGenAIRunner(model_path, execution_provider="cpu")

    # 3. Run inference
    prompt = "Explain quantum computing in simple terms."
    output = runner.generate(prompt, max_length=200, temperature=0.7)

    print("Prompt:", prompt)
    print("Generated:", output)

if __name__ == "__main__":
    main()
