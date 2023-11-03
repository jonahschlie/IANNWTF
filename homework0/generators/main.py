def meow_generator():
    meow_count = 1
    while True:
        meows = " ".join(["Meow"] * meow_count)
        yield f'{meows}'
        meow_count *= 2


gen = meow_generator()

for _ in range(5):
    print(next(gen))
