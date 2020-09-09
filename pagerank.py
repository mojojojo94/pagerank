import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_to_visit_next_page = {}
    number_of_links = len(corpus.get(page))

    if number_of_links == 0:
        equal_probability_to_visit_all_pages_in_corpus = 1 / len(corpus)

        for key in corpus:
            probability_to_visit_next_page.__setitem__(key, equal_probability_to_visit_all_pages_in_corpus)

    else:
        set_of_links = corpus.get(page)
        probability_to_choose_link_on_page = damping_factor / number_of_links
        probability_to_choose_random_page = (1 - damping_factor) / (len(corpus.get(page)) + 1)
        probability_to_visit_next_page.__setitem__(page, probability_to_choose_random_page)  # Add current page to probability

        for link in set_of_links:
            probability_to_visit_next_page.__setitem__(link,
                                                       probability_to_choose_link_on_page + probability_to_choose_random_page)

    return probability_to_visit_next_page


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages_list = []
    page_rank = {}

    random_page = random.choice(list(corpus))
    pages_list.append(random_page)

    first_sample = transition_model(corpus, random_page, damping_factor)
    first_sample_keys = list(first_sample.keys())
    first_sample_values = list(first_sample.values())
    sample_page = random.choices(first_sample_keys, weights=first_sample_values)[0]

    for sample in range(n-1):
        sample_page_distribution = transition_model(corpus, sample_page, damping_factor)
        sample_keys = list(sample_page_distribution.keys())
        sample_values = list(sample_page_distribution.values())
        sample_page = random.choices(sample_keys, weights=sample_values)[0]
        pages_list.append(sample_page)

    for page in pages_list:
        if page in page_rank:
            page_rank[page] += 1
        else:
            page_rank[page] = 1

    page_rank.update((page_name, t_probability / n) for page_name, t_probability in page_rank.items())

    return page_rank


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_rank = {}
    new_page_rank = {}
    equal_probability_to_visit_all_pages_in_corpus = 1 / len(corpus)

    for key in corpus:
        page_rank.__setitem__(key, equal_probability_to_visit_all_pages_in_corpus)

    equal_probability = (1 - damping_factor) / len(corpus)

    not_within_threshold = True

    while not_within_threshold:

        for page in page_rank:
            sum_result = float(0)

            for i in corpus:
                if page in corpus[i]:
                    sum_result += page_rank[i] / len(corpus[i])
                if not corpus[i]:
                    sum_result += page_rank[i] / len(corpus)

            page_rank_of_p = equal_probability + damping_factor * sum_result
            new_page_rank.__setitem__(page, page_rank_of_p)

        not_within_threshold = False

        for page in page_rank:
            if not math.isclose(new_page_rank[page], page_rank[page], abs_tol=0.001):
                not_within_threshold = True
            page_rank[page] = new_page_rank[page]

    return page_rank


if __name__ == "__main__":
    main()
