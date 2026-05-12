import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de Fadiga em Motoristas"
    )
    subparsers = parser.add_subparsers(dest="command", help="Comando disponível")

    prep = subparsers.add_parser("prepare", help="Baixar e preparar dataset")
    prep.add_argument("--dataset", choices=["drowsiness", "mrl", "both"],
                      default="drowsiness", help="Dataset para baixar")

    train = subparsers.add_parser("train", help="Treinar modelo CNN")
    train.add_argument("--model", choices=["mobilenetv2", "cnn"],
                       default="mobilenetv2", help="Arquitetura do modelo")

    detect = subparsers.add_parser("detect", help="Executar detecção em tempo real")
    detect.add_argument("--no-cnn", action="store_true",
                        help="Desativar validação CNN")
    detect.add_argument("--camera", type=int, default=0, help="ID da câmera")
    detect.add_argument("--save", action="store_true", help="Salvar vídeo")
    detect.add_argument("--output", type=str, default=None, help="Caminho do vídeo")

    eval_parser = subparsers.add_parser("evaluate", help="Avaliar modelo")
    eval_parser.add_argument("--full", action="store_true",
                             help="Avaliação completa (CNN + EAR)")

    args = parser.parse_args()

    if args.command == "prepare":
        from src.data_preparation import prepare_dataset
        prepare_dataset(args.dataset)

    elif args.command == "train":
        from src.train_model import train_model
        train_model(args.model)

    elif args.command == "detect":
        from src.detector import DrowsinessDetector
        detector = DrowsinessDetector(
            use_cnn=not args.no_cnn,
            camera_id=args.camera,
        )
        detector.run(
            display=True,
            save_video=args.save,
            output_path=args.output,
        )

    elif args.command == "evaluate":
        from src.evaluate import evaluate_cnn_model, evaluate_ear_on_dataset
        evaluate_cnn_model()
        if args.full:
            evaluate_ear_on_dataset()

    else:
        parser.print_help()
        print("\nExemplos de uso:")
        print("  python main.py prepare --dataset drowsiness")
        print("  python main.py train --model mobilenetv2")
        print("  python main.py detect --camera 0")
        print("  python main.py detect --no-cnn --save")
        print("  python main.py evaluate --full")


if __name__ == "__main__":
    main()
