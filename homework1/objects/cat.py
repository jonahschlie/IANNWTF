class Cat:

    def __init__(self, name: str):
        self.name = name

    def greeting(self, other_cat_name: str) -> None:
        print(f'Hello I am {self.name}! I see you are also a cool '
              f'fluffy kitty {other_cat_name}, let’s together purr at the human, '
              f'so that they shall give us food')
