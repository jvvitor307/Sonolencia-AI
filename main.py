import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Sistema de Detecção de Fadiga em Motoristas"
    )

    parser.add_argument("--camera", type=int, default=0, help="ID da câmera")
    parser.add_argument("--save", action="store_true", help="Salvar vídeo")
    parser.add_argument("--output", type=str, default=None, help="Caminho do vídeo")

    args = parser.parse_args()

    from src.detector import DrowsinessDetector
    detector = DrowsinessDetector(camera_id=args.camera)
    detector.run(
        display=True,
        save_video=args.save,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
