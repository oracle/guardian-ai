.PHONY: build clean install

clean:
	@echo "Cleaning - removing dist, *.pyc, Thumbs.db and other files"
	@rm -rf dist build guardian_ai.egg-info
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;

install:
	@python3 -m pip install .

dist: clean
	@python3 -m build

publish: dist
	@twine upload dist/*