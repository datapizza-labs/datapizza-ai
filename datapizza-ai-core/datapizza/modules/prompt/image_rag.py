from jinja2 import Template

from datapizza.core.modules.prompt import Prompt
from datapizza.core.utils import extract_media
from datapizza.memory import Memory
from datapizza.type import ROLE, Block, Chunk, Media, MediaBlock, TextBlock


class ImageRAGPrompt(Prompt):
    """
    Create a memory for a image RAG system.

    """

    def __init__(
        self,
        user_prompt_template: str,
        image_prompt_presentation: str,
        each_image_template: str,
    ):
        """
        Args:
            user_prompt_template: str # The user prompt jinja template
            image_prompt_presentation: str # The image prompt jinja template
            each_image_template: str # The each image jinja template
        """
        self.user_prompt_template = Template(user_prompt_template)
        self.image_prompt_presentation = image_prompt_presentation
        self.each_image_template = Template(each_image_template)

    def _extract_images_from_chunk(self, chunk: Chunk) -> list[MediaBlock]:
        metadata = chunk.metadata
        list_bboxes = metadata["boundingRegions"]
        path_pdf = metadata["document_name"]

        list_media_blocks: list[MediaBlock] = []

        for bbox in list_bboxes:
            page_number = bbox["pageNumber"]
            polygon = bbox["polygon"]

            # Extract the image from the PDF
            image_as_base64 = extract_media(
                coordinates=polygon,
                file_path=path_pdf,
                page_number=page_number,
            )

            media = Media(
                media_type="image",
                source=image_as_base64,
                source_type="base64",
                extension="png",
            )

            list_media_blocks.append(MediaBlock(media=media))

        return list_media_blocks

    def format(
        self,
        chunks: list[Chunk],
        user_query: str,
        retrieval_query: str,
        memory: Memory | None = None,
    ) -> Memory:
        """
        Creates a new memory object that includes:
        - Existing memory messages
        - User's query
        - Function call retrieval results
        - Chunks retrieval results

        Args:
            chunks: The chunks to add to the memory.
            user_query: The user's query.
            retrieval_query: The query to search the vectorstore for.
            memory: The memory object to add the new messages to.

        Returns:
            memory: A new memory object with the new messages.
        """
        new_memory = Memory()

        if memory:
            for turn in memory:
                new_memory.add_turn(turn.blocks, turn.role)

        all_blocks: list[Block] = [TextBlock(content=self.image_prompt_presentation)]

        for chunk in chunks:
            all_blocks.extend(
                [
                    TextBlock(
                        content=self.each_image_template.render(
                            path_pdf=chunk.metadata["document_name"].split("/")[-1],
                        )
                    )
                ]
            )
            all_blocks.extend(self._extract_images_from_chunk(chunk))

        formatted_user_prompt = self.user_prompt_template.render(
            user_prompt=user_query, retrieval_query=retrieval_query
        )

        all_blocks.append(TextBlock(content=formatted_user_prompt))

        new_memory.add_turn(blocks=all_blocks, role=ROLE.USER)

        return new_memory
