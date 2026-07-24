import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Elimina i file di checkpoint intermedi")
    parser.add_argument("--dir", type=str, default=".", help="Cartella da cui partire (default: corrente)")
    parser.add_argument("--execute", action="store_true", help="Esegue l'eliminazione reale. Senza questo flag, fa solo una simulazione.")
    args = parser.parse_args()

    target_dir = Path(args.dir)
    deleted_count = 0
    freed_space = 0

    print(f"Scansione in corso su: {target_dir.absolute()}...\n")

    for filepath in target_dir.rglob("*epoch*.pt"):
        # Misura di sicurezza: salta il file se contiene la parola "best"
        if "best" in filepath.name.lower():
            print(f"Ignorato (sembra il modello migliore): {filepath.name}")
            continue
            
        size_mb = filepath.stat().st_size / (1024 * 1024)
        
        if not args.execute:
            print(f"[DRY-RUN] Verrebbe eliminato: {filepath} ({size_mb:.2f} MB)")
        else:
            try:
                filepath.unlink()
                print(f"Eliminato: {filepath}")
            except Exception as e:
                print(f"Errore durante l'eliminazione di {filepath}: {e}")
                continue
                
        deleted_count += 1
        freed_space += size_mb

    print("\n--- RESOCONTO ---")
    if not args.execute:
        print("MODALITÀ SIMULAZIONE: Nessun file è stato realmente eliminato.")
        print(f"File trovati: {deleted_count}")
        print(f"Spazio potenziale da liberare: {freed_space / 1024:.2f} GB")
        print("Per eliminare i file, esegui: python cleanup.py --execute")
    else:
        print(f"File eliminati: {deleted_count}")
        print(f"Spazio liberato: {freed_space / 1024:.2f} GB")

if __name__ == "__main__":
    main()