SKILLS_DIR := $(CURDIR)/skills
TARGET_DIR := $(HOME)/.claude/skills

.PHONY: install-skills uninstall-skills list-skills

install-skills:
	@mkdir -p $(TARGET_DIR)
	@for skill in $(SKILLS_DIR)/*/; do \
		name=$$(basename $$skill); \
		ln -sfn $$skill $(TARGET_DIR)/$$name; \
		echo "linked $$name -> $$skill"; \
	done

uninstall-skills:
	@for skill in $(SKILLS_DIR)/*/; do \
		name=$$(basename $$skill); \
		if [ -L $(TARGET_DIR)/$$name ]; then \
			rm $(TARGET_DIR)/$$name; \
			echo "unlinked $$name"; \
		fi; \
	done

list-skills:
	@ls -la $(SKILLS_DIR)/
