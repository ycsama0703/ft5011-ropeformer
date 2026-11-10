"""Cap this process to a fraction of the visible GPU, then run a script as __main__."""
import runpy, sys, torch
torch.cuda.set_per_process_memory_fraction(0.08, 0)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
