CARGO ?= cargo
NAME ?= my_torch_analyzer
DEBUG_BIN := target/debug/$(NAME)
RELEASE_BIN := target/release/$(NAME)

all: release

debug:
	$(CARGO) build
	cp $(DEBUG_BIN) ./$(NAME)

release:
	$(CARGO) build --release
	cp $(RELEASE_BIN) ./$(NAME)

run: debug
	./$(NAME) $(ARGS)

test:
	$(CARGO) test

clean:
	$(CARGO) clean
	@rm -f ./$(NAME)

fclean: clean

re: fclean all

.PHONY: all debug release run clean fclean re test