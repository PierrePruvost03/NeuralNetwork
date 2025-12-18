CARGO ?= cargo
ANALYZER := my_torch_analyzer
GENERATOR := my_torch_generator

DEBUG_ANALYZER := target/debug/$(ANALYZER)
DEBUG_GENERATOR := target/debug/$(GENERATOR)
RELEASE_ANALYZER := target/release/$(ANALYZER)
RELEASE_GENERATOR := target/release/$(GENERATOR)

all: release

debug:
	$(CARGO) build
	cp $(DEBUG_ANALYZER) ./$(ANALYZER)
	cp $(DEBUG_GENERATOR) ./$(GENERATOR)

release:
	$(CARGO) build --release
	cp $(RELEASE_ANALYZER) ./$(ANALYZER)
	cp $(RELEASE_GENERATOR) ./$(GENERATOR)

run: debug
	./$(ANALYZER) $(ARGS)

test:
	$(CARGO) test

clean:
	$(CARGO) clean
	@rm -f ./$(ANALYZER)
	@rm -f ./$(GENERATOR)

fclean: clean

re: fclean all

.PHONY: all debug release run clean fclean re test
