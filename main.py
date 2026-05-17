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

    detect = subparsers.add_parser("detect", help="Executar detecção em tempo real")
    detect.add_argument("--camera", type=int, default=0, help="ID da câmera")
    detect.add_argument("--save", action="store_true", help="Salvar vídeo")
    detect.add_argument("--output", type=str, default=None, help="Caminho do vídeo")

    args = parser.parse_args()

    if args.command == "prepare":
        from src.data_preparation import prepare_dataset
        prepare_dataset(args.dataset)

    elif args.command == "detect":
        from src.detector import DrowsinessDetector
        detector = DrowsinessDetector(camera_id=args.camera)
        detector.run(
            display=True,
            save_video=args.save,
            output_path=args.output,
        )

    else:
        parser.print_help()
        print("\nExemplos de uso:")
        print("  python main.py prepare --dataset drowsiness")
        print("  python main.py detect --camera 0")
        print("  python main.py detect --save")


if __name__ == "__main__":
    main()
