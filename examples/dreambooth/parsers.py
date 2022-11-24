from pathlib import Path
from typing import Tuple

from shared import ParsedConcepts


class ConceptsListParser:
    def parse(self,
              concepts_list: list,
              num_class_images: int,
              with_prior_preservation=True
              ) -> ParsedConcepts:
        inst_img_prompt_tuples = []
        class_img_prompt_tuples = []

        for concept in concepts_list:
            concept_inst_tuples = \
                self._create_tuple_list(concept,
                                        "instance_prompt",
                                        "instance_data_dir")
            inst_img_prompt_tuples.extend(concept_inst_tuples)

            if with_prior_preservation:
                concept_class_tuples = \
                    self._create_tuple_list(concept,
                                            "class_prompt",
                                            "class_data_dir")
                class_img_prompt_tuples.extend(concept_class_tuples[:num_class_images])

        return ParsedConcepts(inst_img_prompt_tuples, class_img_prompt_tuples)

    def _create_tuple_list(self,
                           concept,
                           prompt: str,
                           data_dir: str
                           ) -> list[Tuple]:
        return [(x, concept[prompt]) for x in
                Path(concept[data_dir]).iterdir() if x.is_file()]
