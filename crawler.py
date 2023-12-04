import scrapy

class CrawlingSpider(scrapy.Spider):
    name = "mycrawler"
    allowed_domians = ["mdpi.com"]
    start_urls = ["https://www.mdpi.com"]

    def start_requests(self):
        urls = ["https://www.mdpi.com/journal/sensors"]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def crawlPage(self, response):
        text = response.xpath(
            "//*[@class='hypothesis_container']//div[contains(@class, 'html-body')]//div[contains(@class, 'html-p')]/text()"
        ).extract()
        textFromMPDI = text
        with open('textOfResearchPaper.txt', 'a') as f:
            for item in text:
                f.write(item)

    def parse(self, response):
        # link = response.css("#middle-column").get()
        links = response.xpath(
            "//*[@id='middle-column']//a[contains(@class, 'title-link')]/@href"
        ).getall()
        # print(links)
        for link in links:
            pageLink = "https://www.mdpi.com" + link
            yield response.follow(pageLink, callback=self.crawlPage)
