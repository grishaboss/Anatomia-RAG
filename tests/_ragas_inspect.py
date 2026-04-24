import inspect
from ragas import evaluate, RunConfig

print("evaluate sig:", inspect.signature(evaluate))
rc = RunConfig()
print("RunConfig defaults:", rc)
print("timeout default:", rc.timeout)
print("max_workers default:", rc.max_workers)

# Check executor code
import ragas.executor as ex
print("\nexecutor source snippet:")
src = inspect.getsource(ex)
# Print relevant lines about timeout
for i, line in enumerate(src.split("\n")):
    if "timeout" in line.lower():
        print(f"  {i}: {line}")
