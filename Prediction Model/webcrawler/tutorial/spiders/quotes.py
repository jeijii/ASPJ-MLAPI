import scrapy

class QuotesSpider(scrapy.Spider):
    name = "quotes"
    value = 0
    def start_requests(self):

        # urls = [
        #     'https://www.globalhealingcenter.com/natural-health/cant-sleep-discover-causes-natural-solutions/',
        #     'https://www.globalhealingcenter.com/natural-health/foods-lower-blood-pressure/',
        #     'https://www.globalhealingcenter.com/natural-health/foods-high-in-fiber/',
        #     'https://www.globalhealingcenter.com/natural-health/whats-the-best-colon-cleanse-diet/',
        #     'https://www.globalhealingcenter.com/natural-health/best-laxative-foods/',
        #     'https://www.globalhealingcenter.com/natural-health/benefits-of-prune-juice/',
        #
        #
        #
        # ]

        yield scrapy.Request(url='https://www.globalhealingcenter.com/natural-health/lifestyle/foods', callback = self.parse)


    def parse(self, response):
        number = 0
        length = response.css('div.footer-pagination a::text')[-2].extract()
        for i in range (int(length)):
                number = i + 1
                self.log("Current Number is: " + str(number))
                if number == 1:
                    list = []
                    for item in response.css('a.btn.btn-blue.rounded::attr(href)').extract():
                        yield scrapy.Request(url=item, callback=self.miniMethod)
                else:
                    url = 'https://www.globalhealingcenter.com/natural-health/lifestyle/foods/page/' + str(number)
                    yield scrapy.Request(url=url, callback=self.bigMethod)



    def bigMethod(self , response):

        list = []
        for item in response.css('a.btn.btn-blue.rounded::attr(href)').extract():
            yield scrapy.Request(url=item, callback=self.miniMethod)


    def miniMethod(self, response):
        filename = '%s.txt' %self.value
        datalist = []
        for data in response.css('div.entry-content div p::text').extract():
            datalist.append(data)
        with open(filename , 'wb') as f:
            for d in datalist:
                f.write(bytes(d, 'utf-8'))
        self.log('Saved file %s' % filename)
        self.value  = self.value + 1