const CONTENT = {
	home: {
		title: "Team Information",
		group: "Group13",
		members: [
			"Lê Đức Phương - 2570480",
			"Nguyễn Đình Khánh - 2570227",
			"Nguyễn Huy Hoàng - 2570089",
			"Nguyễn Huỳnh Như - 2570471"
		],
		supervisor: "Dr. Lê Thành Sách"
	},
	assignment: {
		assignment_1: {
			title: "Assignment 1 - Deep Learning for Computer Vision",
			report: "assignment_1/report.pdf",
			readme: "assignment_1/README.md",
			youtube: "https://www.youtube.com/",
			info: [
				"Assignment 1 is organized into 3 independent parts: image classification, text classification, and multimodal classification.",
				"The implementation compares classic deep learning baselines and modern transformer-based approaches.",
				"Each task has its own README, scripts, requirements, and outputs for reproducibility."
			],
			overview: [
				"Task 1: Food-101 image classification with CNN vs ViT, including EDA, augmentation, Grad-CAM, and error analysis.",
				"Task 2: DBpedia-14 text classification with RNN and Transformer, plus CLI and Streamlit inference.",
				"Task 3: CIFAR-100 multimodal classification with OpenCLIP using zero-shot, few-shot, and WiSE-FT."
			],
			tasks: {
				task1: {
					name: "Task 1 - Image Classification",
					directory: "images-classification/food101_project",
					dataset: "Food-101 (Kaggle)",
					objective: "Classify food images into 101 classes and compare CNN-based and Transformer-based models.",
					highlights: [
						"EDA with class distribution, image statistics, and sample visualization.",
						"Data pipeline with 224x224 preprocessing, ImageNet normalization, and augmentation.",
						"Models: ResNet50 and ViT-B/16 with baseline, freeze-backbone, and no-augmentation experiments.",
						"Evaluation with Accuracy, Macro-F1, Precision, Recall, confusion matrix, and top confusions.",
						"Interpretability with Grad-CAM and additional error analysis scripts."
					],
					commands: [
						"cd images-classification/food101_project",
						"pip install -r requirements.txt",
						"python eda.py",
						"python train.py --models resnet50 vit_b_16 --epochs 8 --batch_size 16 --run_name baseline",
						"python run_experiments.py --models resnet50 vit_b_16 --epochs 8 --batch_size 16"
					],
					visuals: [
						{ src: "assignment_1/docs/1.dataset.png", alt: "Food-101 dataset overview" },
						{ src: "assignment_1/docs/1.demo.png", alt: "Image classification demo" }
					]
				},
				task2: {
					name: "Task 2 - Text Classification",
					directory: "text-classification",
					dataset: "DBpedia-14 (HuggingFace)",
					objective: "Build and compare RNN and Transformer models for multi-class text classification.",
					highlights: [
						"Pipeline for text normalization, tokenization, vocabulary building, and padded DataLoader creation.",
						"Config-driven training through config.yml with model-specific hyperparameters.",
						"Training and inference CLI scripts under src/train.py and src/predict.py.",
						"Optional Streamlit app for quick interactive prediction demo.",
						"Model checkpoints saved by architecture for experiment tracking."
					],
					commands: [
						"cd text-classification",
						"pip install -r requirements.txt",
						"python -m src.train --config_file config.yml",
						"python -m src.predict --config_file config.yml --input_text \"This is a test sentence\"",
						"streamlit run streamlit_app.py"
					],
					visuals: [
						{ src: "assignment_1/docs/2.cf_rnn.png", alt: "RNN confusion matrix" },
						{ src: "assignment_1/docs/2.cf_tranf.png", alt: "Transformer confusion matrix" },
						{ src: "assignment_1/docs/2.demotext_demo.png", alt: "Text classification demo" }
					]
				},
				task3: {
					name: "Task 3 - Multimodal Classification",
					directory: "multi-modal-classification",
					dataset: "CIFAR-100 (HuggingFace)",
					objective: "Evaluate OpenCLIP-based multimodal classification with zero-shot, few-shot, and WiSE-FT strategies.",
					highlights: [
						"Zero-shot classification using CLIP text prompts and image-text embedding similarity.",
						"Few-shot linear probing with k-shot sampling across multiple k values.",
						"WiSE-FT interpolation between zero-shot and fine-tuned weights for better robustness.",
						"Standardized evaluation using Accuracy, F1, Precision, and Recall.",
						"Automatic output generation with JSON summaries and visualization plots."
					],
					commands: [
						"cd multi-modal-classification",
						"pip install -r requirements.txt",
						"python run_all.py --mode zero_shot",
						"python run_all.py --mode few_shot",
						"python run_all.py --mode wise_ft"
					],
					visuals: [
						{ src: "assignment_1/docs/3.multi.png", alt: "Multimodal results overview" },
						{ src: "assignment_1/docs/3.OpenCLIP.png", alt: "OpenCLIP pipeline" }
					]
				}
			}
		}
	}
};
