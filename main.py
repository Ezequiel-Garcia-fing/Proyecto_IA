import sys
import subprocess

def mostrar_menu():
    print("\n--- MENÚ PRINCIPAL ---")
    print("1. Reconocimiento de Imágenes")
    print("2. Reconocimiento de Voz")
    print("3. Salir")
    opcion = input("Selecciona una opción (1, 2, 3): ")
    return opcion

def ejecutar_reconocimiento_img():
    print("\nEjecutando reconocimiento de imágenes...\n")
    subprocess.run([sys.executable, "Reconocimiento_IMG.py"])

def ejecutar_reconocimiento_voz():
    print("\nEjecutando reconocimiento de voz...\n")
    subprocess.run([sys.executable, "Reconocimiento_VOZ.py"])

def main():
    while True:
        opcion = mostrar_menu()
        if opcion == "1":
            ejecutar_reconocimiento_img()
        elif opcion == "2":
            ejecutar_reconocimiento_voz()
        elif opcion == "3":
            print("Saliendo del programa.")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
